import asyncio
from adafruit_servokit import ServoKit

class ServoController:
    def __init__(self):
        self.kit = ServoKit(channels=8)
        self.min_angles = [0] * 8
        self.max_angles = [180] * 8
        self.current_angles = [80] * 8
        self.target_angles = [90] * 8
        self.smoothing_factors = [0.3] * 8  # Default smoothing factor for all servos
        self.smoothing_factors[6] = 0.1 # Neck up/down servo smoothing factor
        self.smoothing_factors[7] = 0.1 # Neck left/right servo smoothing factor
        self.enable = False  # Enable flag, set to False at startup
        self.shutdown_flag = False  # Shutdown flag, set to False at startup
        self.minimum_angle_delta = 5  # Minimum angle difference to trigger final angle update
        self.inverted_servos = [4, 5]  # Servos that are inverted in min/max angles
    def set_min_angle(self, servo_index, angle):
        self.min_angles[servo_index] = int(angle)

    def set_max_angle(self, servo_index, angle):
        self.max_angles[servo_index] = int(angle)

    def set_absolute_angle(self, servo_index, angle):
        angle = max(self.min_angles[servo_index], min(self.max_angles[servo_index], int(angle)))
        self.target_angles[servo_index] = angle

    def fractional_to_angle(self, servo_index, fraction):
        if servo_index in self.inverted_servos:
            fraction = 1 - fraction
        angle = self.min_angles[servo_index] + (self.max_angles[servo_index] - self.min_angles[servo_index]) * fraction
        angle = round(angle)
        return angle

    def set_fractional_angle(self, servo_index, fraction):
        if servo_index in self.inverted_servos:
            fraction = 1 - fraction
        angle = self.min_angles[servo_index] + (self.max_angles[servo_index] - self.min_angles[servo_index]) * fraction
        self.target_angles[servo_index] = int(round(angle))
        # print(f"Setting servo {servo_index} to angle {self.target_angles[servo_index]}")
    def set_smoothing_factor(self, servo_index, smoothing_factor):
        self.smoothing_factors[servo_index] = smoothing_factor

    def set_enable(self, enable):
        self.enable = enable

    def get_enable(self):
        return self.enable

    async def move_servo(self, servo_index, delay=0.01):
        while not self.shutdown_flag:
            # print(f"Move servo loop {servo_index}...")
            if self.enable:
                current_angle = self.current_angles[servo_index]
                target_angle = self.target_angles[servo_index]
                smoothing_factor = self.smoothing_factors[servo_index]

                if abs(current_angle - target_angle) > 1:
                    current_angle += (target_angle - current_angle) * smoothing_factor
                    if abs(current_angle - target_angle) < self.minimum_angle_delta:
                        current_angle = target_angle  # Set directly to target to avoid jitter
                    current_angle = int(round(current_angle))
                    # keep current angle within 0 to 180 degrees
                    current_angle = max(0, min(180, current_angle))
                    self.kit.servo[servo_index].angle = current_angle
                    self.current_angles[servo_index] = current_angle
                    # print(f"Moving servo {servo_index} from angle {current_angle} to angle {target_angle}")
                else:
                    self.current_angles[servo_index] = target_angle  # Directly set to target if close

            await asyncio.sleep(delay)

    async def run(self):
        tasks = [self.move_servo(i) for i in range(8)]
        await asyncio.gather(*tasks)

    async def homing(self):
        print("Homing all servos to 0.5 fractional angle...")
        for servo_index in range(8):
            self.set_fractional_angle(servo_index, 0.5)
        print("Homing complete.")

    async def shutdown(self):
        print("Shutting down gracefully...")
        self.set_enable(False)
        self.shutdown_flag = True
        # for servo_index in range(8):
        #     while abs(self.current_angles[servo_index] - self.target_angles[servo_index]) > 1:
        #         await asyncio.sleep(0.01)
        # print("All servos are in stable position. Shutdown complete.")

# Example usage
async def main():
    controller = ServoController()

    # Start the continuous servo update loop
    asyncio.create_task(controller.run())

    # Enable the controller
    controller.set_enable(True)

    # Home all servos to 0.5 fractional angle
    await controller.homing()

    import ipdb; ipdb.set_trace()
    # Example of changing servo angles in real-time
    controller.set_absolute_angle(0, 20)
    await asyncio.sleep(2)
    controller.set_absolute_angle(0, 140)
    await asyncio.sleep(2)
    controller.set_absolute_angle(0, 20)
    await asyncio.sleep(2)

    # Disable the controller
    controller.set_enable(False)

    # Gracefully shut down the controller
    await controller.shutdown()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
