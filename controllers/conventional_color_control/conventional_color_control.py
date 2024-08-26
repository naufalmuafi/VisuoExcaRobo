"""

Conventional Color Control
for Excavator Robot

This controller is used to control the excavator robot to find the target object using the conventional color control method.

by: Naufal Mu'afi

"""

import math
import random
import numpy as np
from controller import Supervisor, Display


MAX_MOTOR_SPEED = 0.7


class ConventionalControl(Supervisor):
    def __init__(self):
        # Initialize the robot instance and timestep
        super(ConventionalControl, self).__init__()
        self.timestep = int(self.getBasicTimeStep())
        random.seed(42)

        # get the robot node
        self.robot = self.getFromDef("EXCAVATOR")

        # set the speed of the motors
        self.max_motor_speed = MAX_MOTOR_SPEED
        self.max_wheel_speed = 7.0

        # set the threshold of the target area
        self.target_threshold = 0.1

        # get the floor node
        arena_tolerance = 1.0
        self.floor = self.getFromDef("FLOOR")
        size_field = self.floor.getField("floorSize").getSFVec3f()

        # set the boundaries of the arena
        x, y = size_field
        self.x_max, self.y_max = x / 2 - arena_tolerance, y / 2 - arena_tolerance
        self.x_min, self.y_min = -self.x_max, -self.y_max

        # initialize camera and display device
        self.camera = self.getDevice("cabin_camera")
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        self.camera.enableRecognitionSegmentation()
        self.display = self.getDevice("segmented_image_display")

        # List of names of the motors and sensors
        names = ["turret", "arm_connector", "lower_arm", "uppertolow", "scoop"]
        wheel = ["lf", "rf", "lb", "rb"]

        # Initialize motors and sensors
        self.wheel_motors = {side: self.getDevice(f"wheel_{side}") for side in wheel}
        self.motors = {name: self.getDevice(f"{name}_motor") for name in names}
        self.sensors = {name: self.getDevice(f"{name}_sensor") for name in names}

        # Configure motor modes
        for motor in list(self.wheel_motors.values()) + list(self.motors.values()):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        # Enable sensors
        for sensor in self.sensors.values():
            sensor.enable(self.timestep)

        # Get the left and right wheel motors
        self.left_wheels = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels = [self.wheel_motors["rf"], self.wheel_motors["rb"]]

    def run(self):
        # Get the camera width and height
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        frame_area = width * height

        # set the boundaries of the target: x-coordinate
        self.center_x = width / 2
        self.tolerance_x = 1.0

        # set the boundaries of the target: y-coordinate
        self.moiety = 2 * height / 3 + 5

        # SOON TO BE DEVELOPED
        # self.step(self.timestep)

        # # set the initial velocity of the turret motor randomly
        # initial_move = random.choice([-1, 1]) * self.max_motor_speed
        # self.motors["turret"].setVelocity(initial_move)

        while self.step(self.timestep) != -1:
            self.state, target_area, centroid = self.get_and_display_obs(
                width, height, frame_area
            )
            done = self.is_done(target_area, self.target_threshold, centroid)

            if done:
                print("sip.")
                # self.digging_operation()
                exit(1)

    def get_and_display_obs(self, width, height, frame_area):
        red_channel, green_channel, blue_channel = [], [], []

        if (
            self.camera.isRecognitionSegmentationEnabled()
            and self.camera.getRecognitionSamplingPeriod() > 0
        ):
            image = self.camera.getImage()
            objects = self.camera.getRecognitionObjects()
            data = self.camera.getRecognitionSegmentationImage()

            if data:
                # Loop through each pixel in the image
                for j in range(height):
                    red_row, green_row, blue_row = [], [], []

                    for i in range(width):
                        # Get the RGB values for the pixel (i, j)
                        red = self.camera.imageGetRed(image, width, i, j)
                        green = self.camera.imageGetGreen(image, width, i, j)
                        blue = self.camera.imageGetBlue(image, width, i, j)

                        # Append the RGB values as a tuple to the row
                        red_row.append(red)
                        green_row.append(green)
                        blue_row.append(blue)

                    # Append the row to the pixels list
                    red_channel.append(red_row)
                    green_channel.append(green_row)
                    blue_channel.append(blue_row)

                # new state
                self.state = np.array(
                    [red_channel, green_channel, blue_channel], dtype=np.uint8
                )

                self.display_segmented_image(data, width, height)

                # calculate the target area
                target_area, centroid = self.recognition_process(
                    self.state, width, height, frame_area
                )

        return self.state, target_area, centroid

    def display_segmented_image(self, data, width, height):
        segmented_image = self.display.imageNew(data, Display.BGRA, width, height)
        self.display.imagePaste(segmented_image, 0, 0, False)
        self.display.imageDelete(segmented_image)

    def recognition_process(self, image, width, height, frame_area):
        target_px = 0
        x_sum = 0
        y_sum = 0

        # Initialize min and max values for x and y
        x_min, x_max = width, 0
        y_min, y_max = height, 0

        # Define color range for red
        lower_red = np.array([200, 0, 0])
        upper_red = np.array([255, 50, 50])

        for y in range(height):
            for x in range(width):
                # get the RGB values for the pixel (x, y)
                r = image[0][y][x]  # Red channel
                g = image[1][y][x]  # Green channel
                b = image[2][y][x]  # Blue channel

                # check if the pixel matches the target color i.e. red color
                if (
                    lower_red[0] <= r <= upper_red[0]
                    and lower_red[1] <= g <= upper_red[1]
                    and lower_red[2] <= b <= upper_red[2]
                ):
                    target_px += 1
                    x_sum += x
                    y_sum += y

                    # Update min and max x and y
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)

        if target_px == 0:
            # No target found, turn over the turret motor
            print("No target found.")

            # set the initial velocity of the turret motor randomly
            initial_move = random.choice([-1, 1]) * self.max_motor_speed
            self.motors["turret"].setVelocity(initial_move)

            return 0, [None, None]

        target_area = target_px / frame_area

        # Calculate the centroid of the target
        centroid_x = x_sum / target_px
        centroid_y = y_sum / target_px
        centroid = [centroid_x, centroid_y]

        # Calculate the current width and height of the target
        target_width = x_max - x_min
        target_height = y_max - y_min

        print(
            f"Centroid: ({centroid_x:.2f}, {centroid_y:.2f}); Target size: {target_width:.1f}x{target_height:.1f}; Target area: {target_area * 100:.2f}%"
        )

        # Move the robot forward if the target is not at the bottom of the frame
        if centroid_y < self.moiety:
            # Move the robot to center_x the target in the frame
            if centroid_x < self.center_x - self.tolerance_x:  # Target is on the left
                if target_area < 0.01:
                    self.turn_left()
                    print("Turning left")
                else:
                    self.motors["turret"].setVelocity(self.max_motor_speed - 0.3)
            elif (
                centroid_x > self.center_x + self.tolerance_x
            ):  # Target is on the right
                if target_area > 0.01:
                    self.turn_right()
                    print("Turning right")
                else:
                    self.motors["turret"].setVelocity(-self.max_motor_speed + 0.3)
            else:
                self.motors["turret"].setVelocity(0.0)
                self.run_wheels(self.max_wheel_speed, "all")

            if target_area < 0.1:
                self.run_wheels(self.max_wheel_speed, "all")
        else:
            self.stop_robot()

        # Move the robot forward if the target is not at the bottom of the frame
        # if centroid_y < self.moiety:
        #     self.run_wheels(self.max_wheel_speed, "all")
        # else:
        #     self.stop_robot()

        return target_area, centroid

    def is_done(self, target_area, threshold=0.25, centroid=[None, None]):
        x_threshold = [
            self.center_x - self.tolerance_x,
            self.center_x + self.tolerance_x,
        ]

        if centroid == [None, None]:
            return False  # No valid centroid found, so not done

        if (target_area >= threshold) or (
            centroid[0] > x_threshold[0]
            and centroid[0] < x_threshold[1]
            and centroid[1] > self.moiety
        ):
            print(
                f"Target area meets or exceeds {threshold * 100:.2f}% of the frame or the centroid is in {centroid}."
            )
            self.run_wheels(0.0, "all")
            self.motors["turret"].setVelocity(0.0)
            return True

        return False

    def run_wheels(self, velocity, wheel="all"):
        if wheel == "all":
            for motor in self.wheel_motors.values():
                motor.setVelocity(velocity)
        elif wheel == "left":
            for motor in self.left_wheels:
                motor.setVelocity(velocity)
        elif wheel == "right":
            for motor in self.right_wheels:
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
    controller = ConventionalControl()
    controller.run()
