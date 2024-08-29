import sys
from typing import Any, Tuple, List
from controller import Supervisor, Display

try:
    import random
    import numpy as np
    import gymnasium as gym
    from gymnasium import Env, spaces
    from gymnasium.envs.registration import EnvSpec, register
except ImportError:
    sys.exit(
        "Please make sure you have all dependencies installed. "
        "Run: `pip install -r requirements.txt`"
    )


MAX_EPISODE_STEPS = 1500
MAX_WHEEL_SPEED = 4.0
MAX_MOTOR_SPEED = 0.7


class Color_VisuoExcaRobo(Supervisor, Env):
    def __init__(self, max_episode_steps: int = MAX_EPISODE_STEPS) -> None:
        # Initialize the Robot class
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        random.seed(42)

        # register the Environment
        self.spec: EnvSpec = EnvSpec(
            id="Color_VisuoExcaRobo", max_episode_steps=max_episode_steps
        )

        # get the robot node
        self.robot = self.getFromDef("EXCAVATOR")

        # set the max_speed of the motors
        self.max_motor_speed = MAX_MOTOR_SPEED
        self.max_wheel_speed = MAX_WHEEL_SPEED

        # set the threshold of the target area
        self.target_threshold = 0.1

        self.floor = self.getFromDef("FLOOR")
        self.set_arena_boundaries()

        self.camera = self.init_camera()
        self.display = self.getDevice("segmented_image_display")

        # Set color range for target detection
        self.lower_color = np.array([35, 42, 44])
        self.upper_color = np.array([55, 62, 64])

        # set the action spaces: 0 = left, 1 = right
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)

        # set the observation space: (channels, camera_height, camera_width)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.camera.getHeight(), self.camera.getWidth()),
            dtype=np.uint8,
        )

        # environment initialization
        self.state = None

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        # Reset the simulation
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        # Set the robot to the initial position
        self.init_pos = self.robot.getPosition()

        # Initialize the motors and sensors
        self.wheel_motors, self.motors, self.sensors = self.init_motors_and_sensors()
        self.left_wheels = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels = [self.wheel_motors["rf"], self.wheel_motors["rb"]]

        # Step of the robot in simulation world
        super().step(self.timestep)

        # Initialize the state
        self.state = np.zeros(
            (3, self.camera.getHeight(), self.camera.getWidth()), dtype=np.uint8
        )

        # info dictionary
        info: dict = {}

        return self.state, info

    def step(self, action):
        width, height = self.camera.getWidth(), self.camera.getHeight()
        frame_area = width * height

        self.center_x = width / 2.0
        self.tolerance_x = 1.0
        self.moiety = 2.0 * height / 3.0 + 5

    def render(self, mode: str = "human") -> Any:
        pass

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
