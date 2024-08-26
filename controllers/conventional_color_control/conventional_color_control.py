"""

Conventional Color Control
for Excavator Robot

This controller is used to control the excavator robot to find the target object using the conventional color control method.

by: Naufal Mu'afi

"""

import math
import random
from controller import Supervisor


class ConventionalControl(Supervisor):
    def __init__(self):
        # Initialize the robot instance and timestep
        super(ConventionalControl, self).__init__()
        self.timeStep = int(self.getBasicTimeStep())
        random.seed(42)

        # set the speed of the motors
        self.max_motor_speed = 0.7
        self.max_wheel_speed = 5.0

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
        self.camera = self.getDevice("camera")
        self.camera.enable(self.timeStep)
        self.camera.recognitionEnable(self.timeStep)
        self.camera.enableRecognitionSegmentation()
        self.display = self.getDevice("segmented_image_display")

        # List of names of the motors and sensors
        names = ["turret", "arm_connector", "lower_arm", "uppertolow", "scoop"]

        # Initialize motors and sensors
        wheel_motors = [self.getDevice(f"wheel{i}_motor") for i in range(1, 5)]
        motors = {name: self.getDevice(f"{name}_motor") for name in names}
        sensors = {name: self.getDevice(f"{name}_sensor") for name in names}

        # Configure motor modes
        for motor in wheel_motors + list(motors.values()):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)

        # Enable sensors
        for sensor in sensors.values():
            sensor.enable(self.timeStep)
