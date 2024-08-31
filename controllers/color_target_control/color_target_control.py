"""

Conventional Color Control
for Excavator Robot

This controller is used to control the excavator robot to find the target object using the conventional color control method.

by: Naufal Mu'afi

"""

import random
import numpy as np
from controller import Supervisor, Display


MAX_MOTOR_SPEED = 0.7


class ColorControl(Supervisor):
    def __init__(self):
        # Initialize the supervisor class
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        random.seed(42)

        # Get the robot node
        self.robot = self.getFromDef("EXCAVATOR")

        # Set the maximum speed for the motors and wheels
        self.max_motor_speed = MAX_MOTOR_SPEED
        self.max_wheel_speed = 4.0
        self.target_threshold = 0.01

        # Get the floor node and set the arena boundaries
        self.floor = self.getFromDef("FLOOR")
        self.set_arena_boundaries()

        # Initialize the camera, motors, and sensors
        self.camera = self.init_camera()
        self.display = self.getDevice("segmented_image_display")
        self.wheel_motors, self.motors, self.sensors = self.init_motors_and_sensors()
        self.left_wheels = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels = [self.wheel_motors["rf"], self.wheel_motors["rb"]]

        # Set the camera properties
        self.camera_width, self.camera_height = (
            self.camera.getWidth(),
            self.camera.getHeight(),
        )
        self.frame_area = self.camera_width * self.camera_height

        # Set the target properties
        self.center_x = self.camera_width / 2
        self.lower_y = self.camera_height
        self.lower_center = [self.center_x, self.lower_y]

        # Set color range for target detection
        color_tolerance = 5
        self.color_target = np.array([46, 52, 54])
        self.lower_color = self.color_target - color_tolerance
        self.upper_color = self.color_target + color_tolerance

        # Set initial move
        self.initial_move = random.choice([0, 1])

    def run(self):
        while self.step(self.timestep) != -1:
            self.state, target_area, centroid = self.get_observation(
                self.camera_width, self.camera_height, self.frame_area
            )
            if self.is_done(target_area, centroid):
                print("sip.")
                # self.digging_operation()
                exit(1)

    def set_arena_boundaries(self):
        arena_tolerance = 1.0
        size_field = self.floor.getField("floorSize").getSFVec3f()
        x, y = size_field
        self.x_max, self.y_max = x / 2 - arena_tolerance, y / 2 - arena_tolerance
        self.x_min, self.y_min = -self.x_max, -self.y_max

    def init_camera(self):
        camera = self.getDevice("cabin_camera")
        camera.enable(self.timestep)
        camera.recognitionEnable(self.timestep)
        camera.enableRecognitionSegmentation()
        return camera

    def init_motors_and_sensors(self):
        names = ["turret", "arm_connector", "lower_arm", "uppertolow", "scoop"]
        wheel = ["lf", "rf", "lb", "rb"]

        wheel_motors = {side: self.getDevice(f"wheel_{side}") for side in wheel}
        motors = {name: self.getDevice(f"{name}_motor") for name in names}
        sensors = {name: self.getDevice(f"{name}_sensor") for name in names}

        for motor in list(wheel_motors.values()) + list(motors.values()):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        for sensor in sensors.values():
            sensor.enable(self.timestep)

        return wheel_motors, motors, sensors

    def get_observation(self, width, height, frame_area):
        if not self.camera.isRecognitionSegmentationEnabled():
            return None, 0, [None, None]

        image = self.camera.getImage()
        data = self.camera.getRecognitionSegmentationImage()
        if not data:
            return None, 0, [None, None]

        red_channel, green_channel, blue_channel = self.extract_rgb_channels(
            image, width, height
        )
        self.state = np.array(
            [red_channel, green_channel, blue_channel], dtype=np.uint8
        )

        self.display_segmented_image(data, width, height)
        return self.state, *self.recognition_process(
            self.state, width, height, frame_area
        )

    def extract_rgb_channels(self, image, width, height):
        red_channel, green_channel, blue_channel = [], [], []
        for j in range(height):
            red_row, green_row, blue_row = [], [], []
            for i in range(width):
                red_row.append(self.camera.imageGetRed(image, width, i, j))
                green_row.append(self.camera.imageGetGreen(image, width, i, j))
                blue_row.append(self.camera.imageGetBlue(image, width, i, j))
            red_channel.append(red_row)
            green_channel.append(green_row)
            blue_channel.append(blue_row)
        return red_channel, green_channel, blue_channel

    def display_segmented_image(self, data, width, height):
        segmented_image = self.display.imageNew(data, Display.BGRA, width, height)
        self.display.imagePaste(segmented_image, 0, 0, False)
        self.display.imageDelete(segmented_image)

    def recognition_process(self, image, width, height, frame_area):
        target_px, target_area = 0, 0
        x_min, x_max, y_min, y_max = width, 0, height, 0

        for y in range(height):
            for x in range(width):
                r, g, b = image[0][y][x], image[1][y][x], image[2][y][x]
                if (
                    self.lower_color[0] <= r <= self.upper_color[0]
                    and self.lower_color[1] <= g <= self.upper_color[1]
                    and self.lower_color[2] <= b <= self.upper_color[2]
                ):
                    target_px += 1
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    y_min, y_max = min(y_min, y), max(y_max, y)

        if target_px == 0:
            self.search_target()
            return 0, [None, None]

        target_area = target_px / frame_area
        centroid = [(x_max + x_min) / 2, (y_max + y_min) / 2]

        print(
            f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}); Target size: {x_max - x_min:.1f}x{y_max - y_min:.1f}; Target area: {target_area * 100:.2f}%"
        )
        self.move_towards_target(centroid, target_area)
        return target_area, centroid

    def search_target(self):
        print("No target found.")

        if self.initial_move == 0:
            self.run_wheels(self.initial_move, "left")
        elif self.initial_move == 1:
            self.run_wheels(-self.initial_move, "right")

    def move_towards_target(self, centroid, target_area):
        if (centroid[1] < self.moiety or target_area < 0.01) or (
            self.center_x - self.tolerance_x
            <= centroid[0]
            <= self.center_x + self.tolerance_x
        ):
            if centroid[0] <= self.center_x - self.tolerance_x:
                self.adjust_turret_and_wheels(target_area, direction="left")
                print("Adjusting turret and wheels to the left.")
            elif centroid[0] >= self.center_x + self.tolerance_x:
                self.adjust_turret_and_wheels(target_area, direction="right")
                print("Adjusting turret and wheels to the right.")
            else:
                self.motors["turret"].setVelocity(0.0)
                self.run_wheels(self.max_wheel_speed, "all")
                print("Moving forward.")
        else:
            self.stop_robot()

    def adjust_turret_and_wheels(self, target_area, direction):
        self.motors["turret"].setVelocity(0.0)
        if target_area < 0.009:
            if direction == "left":
                self.turn_left()
            elif direction == "right":
                self.turn_right()
        else:
            turret_speed = (
                self.max_motor_speed - 0.3
                if direction == "left"
                else -self.max_motor_speed + 0.3
            )
            self.motors["turret"].setVelocity(turret_speed)
            self.run_wheels(self.max_wheel_speed, "all")

    def is_done(self, target_area, centroid):
        if centroid == [None, None]:
            return False

        x_threshold = [
            self.center_x - self.tolerance_x,
            self.center_x + self.tolerance_x,
        ]
        if target_area >= self.target_threshold and (
            x_threshold[0] <= centroid[0] <= x_threshold[1]
            and centroid[1] > self.moiety
        ):
            print(
                f"Target area meets or exceeds {self.target_threshold * 100:.2f}% of the frame or the centroid is in {centroid}."
            )
            self.stop_robot()
            return True
        return False

    def run_wheels(self, velocity, wheel="all"):
        wheels = (
            self.left_wheels + self.right_wheels
            if wheel == "all"
            else self.left_wheels if wheel == "left" else self.right_wheels
        )
        for motor in wheels:
            motor.setVelocity(velocity)

    def turn_left(self):
        self.run_wheels(-self.max_wheel_speed, "left")
        self.run_wheels(self.max_wheel_speed, "right")

    def turn_right(self):
        self.run_wheels(self.max_wheel_speed, "left")
        self.run_wheels(-self.max_wheel_speed, "right")

    def stop_robot(self):
        self.run_wheels(0.0, "all")

    # 0 is left, 1 is right
    def move_arm_connector(
        self,
        direction,
        min_position=-1.1,
        max_position=1.1,
        velocity=MAX_MOTOR_SPEED,
        toCenter=False,
    ):
        current_position = self.sensors["arm_connector"].getValue()

        if toCenter:
            tolerance = 0.001
            if current_position > tolerance or current_position < -tolerance:
                self.motors["arm_connector"].setVelocity(
                    velocity * (1 if current_position < 0 else -1)
                )
            elif current_position < tolerance or current_position > -tolerance:
                self.motors["arm_connector"].setVelocity(0.0)
        else:
            # Check if the motor is within the defined range
            if min_position <= current_position <= max_position:
                self.motors["arm_connector"].setVelocity(
                    velocity * (1 if direction == 0 else -1)
                )
            else:
                self.motors["arm_connector"].setVelocity(0.0)

    # 0 is down, 1 is up
    def move_lower_arm(
        self, direction, min_position=-0.27, max_position=0.27, velocity=MAX_MOTOR_SPEED
    ):
        current_position = self.sensors["lower_arm"].getValue()

        # Check if the motor is within the defined range
        if min_position <= current_position <= max_position:
            self.motors["lower_arm"].setVelocity(
                velocity * (1 if direction == 0 else -1)
            )
        else:
            self.motors["lower_arm"].setVelocity(0.0)

    # 0 is down, 1 is up
    def move_uppertolow(
        self, direction, min_position=-0.9, max_position=0.9, velocity=MAX_MOTOR_SPEED
    ):
        current_position = self.sensors["uppertolow"].getValue()

        # Check if the motor is within the defined range
        if min_position <= current_position <= max_position:
            self.motors["uppertolow"].setVelocity(
                velocity * (1 if direction == 0 else -1)
            )
        else:
            self.motors["uppertolow"].setVelocity(0.0)

    # 0 is inside, 1 is outside
    def move_scoop(
        self,
        direction,
        min_position=-1.1,
        max_position=1.1,
        velocity=MAX_MOTOR_SPEED + 0.5,
    ):
        current_position = self.sensors["scoop"].getValue()

        # Check if the motor is within the defined range
        if min_position <= current_position <= max_position:
            self.motors["scoop"].setVelocity(velocity * (1 if direction == 0 else -1))
        else:
            self.motors["scoop"].setVelocity(0.0)

    def digging_operation(self):
        initial_positions = {
            "scoop": self.sensors["scoop"].getValue(),
            "lower_arm": self.sensors["lower_arm"].getValue(),
            "uppertolow": self.sensors["uppertolow"].getValue(),
            "arm_connector": self.sensors["arm_connector"].getValue(),
        }

        targets = {
            "scoop": 1.0,
            "lower_arm": 0.1,
            "uppertolow": 0.45,
        }

        step = 0
        delay_start_time = None

        while True:
            current_positions = {
                "scoop": self.sensors["scoop"].getValue(),
                "lower_arm": self.sensors["lower_arm"].getValue(),
                "uppertolow": self.sensors["uppertolow"].getValue(),
            }

            if step == 0:
                # Adjust the target for uppertolow
                adjusted_targets = {
                    "scoop": -targets["scoop"],
                    "lower_arm": -targets["lower_arm"],
                    "uppertolow": -targets["uppertolow"],
                }

                self.move_scoop(
                    0,
                    min_position=initial_positions["scoop"],
                    max_position=adjusted_targets["scoop"],
                )
                self.move_lower_arm(
                    0,
                    min_position=initial_positions["lower_arm"],
                    max_position=adjusted_targets["lower_arm"],
                )
                self.move_uppertolow(1, min_position=adjusted_targets["uppertolow"])
                self.move_arm_connector(1, toCenter=True)

                print(f"Current positions: {current_positions}")
                print(f"Adjusted targets: {adjusted_targets}")

                # Check if all the joints have reached or exceeded their adjusted targets
                if all(
                    current_positions[joint] >= adjusted_targets[joint]
                    for joint in adjusted_targets
                ):
                    delay_start_time = self.getTime()
                    print("Step 0 done.")
                    step = 1

            elif step == 1:
                print("Step 1")
                if self.getTime() - delay_start_time >= 2.0:
                    step = 2

            elif step == 2:
                print("Step 2")
                # Move down
                self.move_scoop(0, max_position=targets["scoop"] - 0.5)
                self.move_lower_arm(0, max_position=targets["lower_arm"] - 0.1)
                self.move_uppertolow(0, max_position=targets["uppertolow"])
                self.move_arm_connector(1, toCenter=True)

                adjusted_targets = {
                    "lower_arm": targets["lower_arm"] - 0.1,
                    "scoop": targets["scoop"] - 0.5,
                }
                # Check if all the joints have reached or exceeded their adjusted targets
                if all(
                    current_positions[joint] <= adjusted_targets.get(joint, target)
                    for joint, target in adjusted_targets.items()
                ):
                    print("Step 2 done.")
                    delay_start_time = self.getTime()
                    step = 3

            elif step == 3:
                if self.getTime() - delay_start_time >= 1.0:
                    return [
                        initial_positions[joint]
                        for joint in [
                            "arm_connector",
                            "lower_arm",
                            "uppertolow",
                            "scoop",
                        ]
                    ]

            self.step(self.timestep)


if __name__ == "__main__":
    controller = ColorControl()
    controller.run()
