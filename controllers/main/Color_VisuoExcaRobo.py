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
MAX_ROBOT_DISTANCE = 0.7


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

        # Get the camera width and height
        self.width, self.height = self.camera.getWidth(), self.camera.getHeight()
        self.frame_area = self.width * self.height

        # Set the target position with intialize the threshold
        self.center_x = self.width / 2.0
        self.tolerance_x = 1.0
        self.moiety = 2.0 * self.height / 3.0 + 5

        # Set color range for target detection
        self.lower_color = np.array([35, 42, 44])
        self.upper_color = np.array([55, 62, 64])

        # set the action spaces: 0 = left, 1 = right
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

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
        # Get the observation to calculate target_area
        self.state, target_area, centroid = self.get_observation(
            self.width, self.height, self.frame_area
        )

        if target_area <= 0.01:
            # Prioritize moving wheels
            left_wheels_action = action[0] * self.max_wheel_speed
            right_wheels_action = action[1] * self.max_wheel_speed
            turret_action = 0.0  # Turret doesn't move when target_area is too small
        else:
            # Allow turret movement
            left_wheels_action = action[0] * self.max_wheel_speed
            right_wheels_action = action[1] * self.max_wheel_speed
            turret_action = action[2] * self.max_motor_speed

        # Control the wheels
        self.run_wheels(left_wheels_action, "left")
        self.run_wheels(right_wheels_action, "right")

        # Control the turret motor
        self.motors["turret"].setVelocity(turret_action)

        # Go to the next step
        super().step(self.timestep)

        # Get the new observation and reward after action application
        self.state, target_area, centroid = self.get_observation(
            self.width, self.height, self.frame_area
        )
        reward = self.compute_reward(target_area, centroid)

        # Determine if the episode is done
        done = self.check_done(target_area, centroid)

        return self.state, reward, done, False, {}

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

    def recognition_process(self, image, width, height, frame_area):
        target_px, x_sum, y_sum = 0, 0, 0
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
                    x_sum += x
                    y_sum += y
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    y_min, y_max = min(y_min, y), max(y_max, y)

        if target_px == 0:
            return 0, [None, None]

        target_area = target_px / frame_area
        centroid = [x_sum / target_px, y_sum / target_px]

        return target_area, centroid

    def run_wheels(self, velocity, wheel):
        wheels = self.left_wheels if wheel == "left" else self.right_wheels
        for motor in wheels:
            motor.setVelocity(velocity)

    def compute_reward(self, target_area, centroid):
        reward = 0

        # Reward or punishment based on the change in target area
        previous_target_area = getattr(self, "previous_target_area", 0)
        area_increase = target_area - previous_target_area

        if area_increase > 0:
            reward += 50  # Reward for progress
        else:
            reward -= 50  # Penalty for stagnation or moving away from target

        # Update previous target area for the next step
        self.previous_target_area = target_area

        # Reward based on pixel position (x and y alignment)
        if centroid == [None, None]:
            reward -= 100  # Penalty if the target is not visible
        else:
            x_error = abs(centroid[0] - self.center_x)
            reward -= x_error * 10  # Penalize for being far from center

            # Additional reward if the target is within the x-coordinate threshold
            x_threshold = [
                self.center_x - self.tolerance_x,
                self.center_x + self.tolerance_x,
            ]

            if x_threshold[0] <= centroid[0] <= x_threshold[1]:
                reward += 10000  # Big reward for x-coordinate alignment

            if centroid[1] >= self.moiety:
                reward += 10000  # Reward for being below the moiety
            else:
                reward -= (
                    self.moiety - centroid[1]
                ) * 10  # Penalize for being too high

        # Additional reward for reaching or exceeding the target area threshold
        if target_area >= self.target_threshold:
            reward += 10000  # Big reward for reaching the target area threshold

        # Penalty if the robot is too far from the starting position
        pos = self.robot.getPosition()
        distance = (
            (pos[0] - self.init_pos[0]) ** 2 + (pos[1] - self.init_pos[1]) ** 2
        ) ** 0.5
        if distance > MAX_ROBOT_DISTANCE:
            reward -= 10000  # Significant penalty for being too far

        # Penalty for hitting arena boundaries
        if (
            pos[0] <= self.x_min
            or pos[0] >= self.x_max
            or pos[1] <= self.y_min
            or pos[1] >= self.y_max
        ):
            reward -= 10000  # Significant penalty for hitting the boundary
            return reward  # Mark episode as done

        # Apply a small time penalty to encourage quicker completion
        reward -= 1

        return reward

    def check_done(self, target_area, centroid):
        # End episode if the target area is reached
        if target_area >= self.target_threshold:
            return True

        distance = (
            (pos[0] - self.init_pos[0]) ** 2 + (pos[1] - self.init_pos[1]) ** 2
        ) ** 0.5
        if distance > MAX_ROBOT_DISTANCE:
            return True

        # End episode if the robot has hit the arena boundaries
        pos = self.robot.getPosition()
        if (
            pos[0] <= self.x_min
            or pos[0] >= self.x_max
            or pos[1] <= self.y_min
            or pos[1] >= self.y_max
        ):
            return True

        return False
