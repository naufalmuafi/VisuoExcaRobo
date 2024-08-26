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


class ConventionalControl(Supervisor):
    def __init__(self):
        # Initialize the robot instance and timestep
        super(ConventionalControl, self).__init__()
        self.timestep = int(self.getBasicTimeStep())
        random.seed(42)

        # set the speed of the motors
        self.max_motor_speed = 0.7
        self.max_wheel_speed = 7.0

        # set the threshold of the target area
        self.target_threshold = 0.35

        # get the robot node
        self.robot = self.getFromDef("EXCAVATOR")

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

        # SOON TO BE DEVELOPED
        # self.step(self.timestep)

        # # set the initial velocity of the turret motor randomly
        # initial_move = random.choice([-1, 1]) * self.max_motor_speed
        # self.motors["turret"].setVelocity(initial_move)

        while self.step(self.timestep) != -1:
            self.state, target_area = self.get_and_display_obs(
                width, height, frame_area
            )
            done = self.is_done(target_area, self.target_threshold)

            if done:
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
                target_area = self.recognition_process(
                    self.state, width, height, frame_area
                )

            # self.objects_recognition(objects, width, target_area)

        return self.state, target_area

    def display_segmented_image(self, data, width, height):
        segmented_image = self.display.imageNew(data, Display.BGRA, width, height)
        self.display.imagePaste(segmented_image, 0, 0, False)
        self.display.imageDelete(segmented_image)

    def recognition_process(self, image, width, height, frame_area):
        target_px = 0
        x_sum = 0
        y_sum = 0

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

        if target_px == 0:
            # No target found, stop the robot or continue searching
            print("No target found.")
            return 0

        target_area = target_px / frame_area

        # Calculate the centroid of the target
        target_x = x_sum / target_px
        target_y = y_sum / target_px

        # Move the robot to center the target in the frame
        if target_x < width / 3:
            self.motors["turret"].setVelocity(self.max_motor_speed - 0.3)
        elif target_x > 2 * width / 3:
            self.motors["turret"].setVelocity(-self.max_motor_speed + 0.3)
        else:
            self.motors["turret"].setVelocity(0.0)

        # Move the robot forward if the target is not at the bottom of the frame
        if target_y < 2 * height / 3:
            self.run_wheels(self.max_wheel_speed, "all")
        else:
            self.stop_robot()

        return target_area

    def objects_recognition(self, objects, width, target_area):
        for obj in objects:
            for i in range(obj.getNumberOfColors()):
                r, g, b = obj.getColors()[3 * i : 3 * i + 3]
                print(f"Color {i + 1}/{obj.getNumberOfColors()}: {r} {g} {b}")

                if r == 1 and g == 0 and b == 0:
                    print("Target found, determining position...")
                    print(f"Target area: {target_area*100:.2f}% of the frame")

                    position_on_image = obj.getPositionOnImage()
                    obj_x, obj_y = position_on_image[0], position_on_image[1]

                    print(f"Object position on image: x={obj_x}, y={obj_y}")

                    if obj_x < width / 3:
                        self.turn_left()
                    elif obj_x < 2 * width / 3:
                        self.run_wheels(self.max_wheel_speed, "all")
                    else:
                        self.turn_right()

    def is_done(self, target_area, threshold=0.25):
        if target_area >= threshold:
            print(f"Target area meets or exceeds {threshold * 100:.2f}% of the frame.")
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
        print("Robot stopped.")


if __name__ == "__main__":
    controller = ConventionalControl()
    controller.run()
