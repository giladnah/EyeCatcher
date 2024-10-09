import time
import asyncio
import random
from multiprocessing import Manager, Process
from ControllerH import ServoController

# Operation description:
# Servos:
# 0 - control Eyes X axis (0 right, 1 left)
# 1 - control Eyes Y axis (0 down, 1 up)
# 2 - control Right top eyelid (0 down, 1 up)
# 3 - control Right bottom eyelid (0 down, 1 up)
# 4 - control Left top eyelid (0 up, 1 down) - Inverted in min/max
# 5 - control Left bottom eyelid (0 up, 1 down) - Inverted in min/max
# 6 - control neck up/down
# 7 - control neck left/right

class EyeDataController:
    def __init__(self, shared_data, shared_config, shared_lock, debug=False):
        self.shared_data = shared_data
        self.shared_config = shared_config
        self.shared_lock = shared_lock
        self.debug = debug
        self.shutdown_flag = False
        self.blink = False
        # Create an instance of the ServoController
        self.servo_controller = ServoController()
        # Set the minimum and maximum angles for each servo
        self.update_servo_config()


    async def run(self):
        print("Running EyeDataController...")
        tasks = []
        tasks.append(asyncio.create_task(self.servo_controller.run()))
        self.set_enable(True)
        await self.servo_controller.homing()
        # if self.shared_data['auto_blink']:
        tasks.append(asyncio.create_task(self.auto_blink()))
        tasks.append(asyncio.create_task(self.update_controller()))
        await asyncio.gather(*tasks)

    async def update_controller(self):
        while not self.shutdown_flag:
            self.update_servo_config()  # Update config dynamically TBD should not be done in the loop
            await self.update_control_state()
            auto_eyelids = self.shared_data['auto_eyelids']
            for servo, config in self.shared_config.items():
                if 'eyelid' in servo and auto_eyelids:
                    if self.blink:
                    # if True:
                        self.servo_controller.set_absolute_angle(config['index'], 90)
                    else:
                        self.servo_controller.set_fractional_angle(config['index'], self.shared_data['eyes_y'])
                else:
                    self.servo_controller.set_fractional_angle(config['index'], self.shared_data[servo])
            await asyncio.sleep(0.03)

    def update_servo_config(self):
        for servo, config in self.shared_config.items():
            self.servo_controller.set_min_angle(config['index'], config['min'])
            self.servo_controller.set_max_angle(config['index'], config['max'])

    async def update_control_state(self):
        self.set_enable(self.shared_data['enable'])
        if self.shared_data['shutdown_flag'] and not self.shutdown_flag:
            self.shutdown_flag = True
            print("Shutting down EyeDataController...")
            await self.close()

    def set_enable(self, enable):
        self.shared_data['enable'] = enable
        self.servo_controller.set_enable(enable)

    async def auto_blink(self):
        while not self.shutdown_flag:
            if self.shared_data['auto_blink']:
                await asyncio.sleep(random.uniform(self.shared_data['blink_min_period'], self.shared_data['blink_max_period']))
                # Close eyelids
                print("Blinking...")
                self.blink = True
                # self.set_fractional_servo_angles({servo: 0.0 for servo in self.shared_config if 'eyelid' in servo})
                await asyncio.sleep(self.shared_data['blink_period'])  # Short pause to simulate blink
                # Open eyelids to previous state
                print("Opening eyes...")
                self.blink = False
            else:
                await asyncio.sleep(1)

    def set_fractional_servo_angles(self, fractions):
        for servo, fraction in fractions.items():
            config = self.shared_config[servo]
            self.servo_controller.set_fractional_angle(config['index'], fraction)

    async def close(self):
        self.set_enable(False)
        await self.servo_controller.shutdown()

    @staticmethod
    def get_shared_data_structure():
        return {
            'eyes_x': 0.5,
            'eyes_y': 0.5,
            'right_top_eyelid': 0.5,
            'right_bottom_eyelid': 0.5,
            'left_top_eyelid': 0.5,
            'left_bottom_eyelid': 0.5,
            'neck_up_down': 0.5,
            'neck_left_right': 0.5,
            'auto_blink': False,
            'blink_min_period': 2,
            'blink_max_period': 5,
            'blink_period': 0.2,
            'enable': False,
            'shutdown_flag': False,
            'auto_eyelids': True # Automatically control eyelids based on and eyes_y
        }

    @staticmethod
    def get_shared_config_structure():
        return {
            'eyes_x': {'index': 0, 'min': 45, 'max': 135},
            'eyes_y': {'index': 1, 'min': 45, 'max': 135},
            'right_top_eyelid': {'index': 2, 'min': 90, 'max': 180},
            'right_bottom_eyelid': {'index': 3, 'min': 0, 'max': 90},
            'left_top_eyelid': {'index': 4, 'min': 0, 'max': 90}, # Inverted
            'left_bottom_eyelid': {'index': 5, 'min': 90, 'max': 180}, # Inverted
            'neck_up_down': {'index': 6, 'min': 45, 'max': 135},
            'neck_left_right': {'index': 7, 'min': 45, 'max': 135}
        }

async def eye_controller_main(eye_controller):
    tasks = []
    tasks.append(asyncio.create_task(eye_controller.run()))
    await asyncio.gather(*tasks)
    # await asyncio.sleep(2)
    # shared_data['eyes_x'] = 0.2
    # await asyncio.sleep(2)
    # shared_data['eyes_x'] = 0.8
    # await asyncio.sleep(10)
    # await eye_controller.close()

def run_eye_controller(shared_data, shared_config, shared_lock):
    eye_controller = EyeDataController(shared_data, shared_config, shared_lock, debug=True)
    asyncio.run(eye_controller_main(eye_controller))

if __name__ == "__main__":
    with Manager() as manager:
        shared_lock = manager.Lock()
        shared_data = manager.dict(EyeDataController.get_shared_data_structure())
        shared_config = manager.dict(EyeDataController.get_shared_config_structure())
        eye_process = Process(target=run_eye_controller, args=(shared_data, shared_config, shared_lock))
        eye_process.start()
        while True:
            import ipdb; ipdb.set_trace()
            time.sleep(1)
        try:
            for i in range(10):
                # Your main application code here
                # Example: Updating shared data to control EyeDataController
                with shared_lock:
                    shared_data['eyes_x'] = 0.5
                    shared_data['eyes_y'] = 0.5
                time.sleep(1)
                with shared_lock:
                    shared_data['eyes_x'] = 0.0
                    shared_data['eyes_y'] = 0.0
                time.sleep(1)
                with shared_lock:
                    shared_data['eyes_x'] = 1.0
                    shared_data['eyes_y'] = 0.0
                time.sleep(1)
                with shared_lock:
                    shared_data['eyes_x'] = 0.0
                    shared_data['eyes_y'] = 1.0
                time.sleep(1)
                with shared_lock:
                    shared_data['eyes_x'] = 1.0
                    shared_data['eyes_y'] = 1.0
            shared_data['shutdown_flag'] = True
        except KeyboardInterrupt:
            pass
        finally:
            eye_process.terminate()
            eye_process.join()
