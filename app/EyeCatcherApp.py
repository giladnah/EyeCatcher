import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
from multiprocessing import Manager, Process
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)
import EyeController

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self, shared_data, shared_config, shared_lock):
        super().__init__()
        self.shared_data = shared_data
        self.shared_config = shared_config
        self.shared_lock = shared_lock
# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # return Gst.PadProbeReturn.OK
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    for detection in detections:
        # import ipdb; ipdb.set_trace()
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person":
            string_to_print += (f"Detection: {label} {confidence:.2f}\n")
            # Pose estimation landmarks from detection (if available)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                left_eye = points[1]  # assuming 1 is the index for the left eye
                right_eye = points[2]  # assuming 2 is the index for the right eye
                nose = points[0]
                # The landmarks are normalized to the bounding box, we also need to convert them to the frame size
                # left_eye_x = int((left_eye.x() * bbox.width() + bbox.xmin()) * width)
                # left_eye_y = int((left_eye.y() * bbox.height() + bbox.ymin()) * height)
                # right_eye_x = int((right_eye.x() * bbox.width() + bbox.xmin()) * width)
                # right_eye_y = int((right_eye.y() * bbox.height() + bbox.ymin()) * height)
                nose_x = (nose.x() * bbox.width() + bbox.xmin())
                nose_y = (nose.y() * bbox.height() + bbox.ymin())
                with user_data.shared_lock:
                    user_data.shared_data['eyes_x'] = (nose_x)
                    # user_data.shared_data['eyes_y'] = 1 - (nose_y)
                    # map nose_y from 0.3 to 0.7 to 1 to 0
                    user_data.shared_data['eyes_y'] = 1- ((nose_y - 0.3) / (0.7 - 0.3))

                    print(f"eyes_x: {user_data.shared_data['eyes_x']}, eyes_y: {user_data.shared_data['eyes_y']}")

    # print(string_to_print)
    return Gst.PadProbeReturn.OK



# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
        'nose': 1,
        'left_eye': 2,
        'right_eye': 3,
        'left_ear': 4,
        'right_ear': 5,
        'left_shoulder': 6,
        'right_shoulder': 7,
        'left_elbow': 8,
        'right_elbow': 9,
        'left_wrist': 10,
        'right_wrist': 11,
        'left_hip': 12,
        'right_hip': 13,
        'left_knee': 14,
        'right_knee': 15,
        'left_ankle': 16,
        'right_ankle': 17,
    }

    return keypoints
#-----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class

class GStreamerPoseEstimationApp(GStreamerApp):
    def __init__(self, args, user_data):
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolov8pose_post.so')
        self.post_function_name = "filter"
        self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_pose_h8l_pi.hef')
        self.app_callback = app_callback

        # Set the process title
        setproctitle.setproctitle("Hailo Pose Estimation App")

        self.create_pipeline()

    def get_pipeline_string(self):
        class_id = 0 # class_id for person
        TRACKER = f'hailotracker name=hailo_tracker class-id={class_id} kalman-dist-thr=0.8 iou-thr=0.9 init-iou-thr=0.7 \
                keep-new-frames=2 keep-tracked-frames=15 keep-lost-frames=2 keep-past-metadata=true qos=false ! '
        if self.source_type == "rpi":
            source_element = f"libcamerasrc name=src_0 ! "
            source_element += f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
            source_element += QUEUE("queue_src_scale")
            source_element += f"videoscale ! "
            source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "
            source_element += 'videoflip video-direction=vert ! '
        elif self.source_type == "usb":
            source_element = f"v4l2src device={self.video_source} name=src_0 ! "
            source_element += f"video/x-raw, width=640, height=480, framerate=30/1 ! "
            source_element += 'videoflip video-direction=horiz ! '
        else:
            source_element = f"filesrc location=\"{self.video_source}\" name=src_0 ! "
            source_element += QUEUE("queue_dec264")
            source_element += f" qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
            source_element += f" video/x-raw,format=I420 ! "
        source_element += QUEUE("queue_scale")
        source_element += f"videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert")
        source_element += f"videoconvert n-threads=3 name=src_convert qos=false ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "

        pipeline_string = "hailomuxer name=hmux "
        pipeline_string += source_element
        pipeline_string += "tee name=t ! "
        pipeline_string += QUEUE("bypass_queue", max_size_buffers=20) + "hmux.sink_0 "
        pipeline_string += "t. ! " + QUEUE("queue_hailonet")
        pipeline_string += "videoconvert n-threads=3 ! "
        pipeline_string += f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} force-writable=true ! "
        pipeline_string += QUEUE("queue_hailofilter")
        pipeline_string += f"hailofilter function-name={self.post_function_name} so-path={self.default_postprocess_so} qos=false ! "
        pipeline_string += QUEUE("queue_hmuc") + " hmux.sink_1 "
        pipeline_string += "hmux. ! " + QUEUE("queue_hailo_python")
        pipeline_string += TRACKER
        pipeline_string += QUEUE("queue_user_callback")
        pipeline_string += f"identity name=identity_callback ! "
        pipeline_string += QUEUE("queue_hailooverlay")
        pipeline_string += f"hailooverlay ! "
        pipeline_string += QUEUE("queue_videoconvert")
        pipeline_string += f"videoconvert n-threads=3 qos=false ! "
        pipeline_string += QUEUE("queue_hailo_display")
        pipeline_string += f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Define Manager and keep it active
    with Manager() as manager:
        shared_lock = manager.Lock()
        shared_data = manager.dict(EyeController.EyeDataController.get_shared_data_structure())
        shared_config = manager.dict(EyeController.EyeDataController.get_shared_config_structure())
        eye_process = Process(target=EyeController.run_eye_controller, args=(shared_data, shared_config, shared_lock))
        eye_process.start()
        # Create an instance of the user app callback class
        user_data = user_app_callback_class(shared_data, shared_config, shared_lock)
        parser = get_default_parser()
        args = parser.parse_args()
        app = GStreamerPoseEstimationApp(args, user_data)
        app.run()
