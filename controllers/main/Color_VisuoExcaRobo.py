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


ENV_ID = "Color_VisuoExcaRobo"
MAX_EPISODE_STEPS = 3000
MAX_WHEEL_SPEED = 5.0
MAX_MOTOR_SPEED = 0.7
MAX_ROBOT_DISTANCE = 13.0
DISTANCE_THRESHOLD = 1.0
LOWER_Y = -19


class Color_VisuoExcaRobo(Supervisor, Env):
    def __init__(self, max_episode_steps: int = MAX_EPISODE_STEPS) -> None:
        # Initialize the Robot class
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        random.seed(42)

        # register the Environment
        self.spec: EnvSpec = EnvSpec(id=ENV_ID, max_episode_steps=max_episode_steps)

        # get the robot node
        self.robot = self.getFromDef("EXCAVATOR")

        # set the max_speed of the motors
        self.max_motor_speed = MAX_MOTOR_SPEED
        self.max_wheel_speed = MAX_WHEEL_SPEED
        self.max_robot_distance = MAX_ROBOT_DISTANCE
        self.distance_threshold = DISTANCE_THRESHOLD

        # Get the floor node and set the arena boundaries
        self.floor = self.getFromDef("FLOOR")
        self.set_arena_boundaries()

        # Initialize the camera, and displays (optional)
        self.camera = self.init_camera()
        self.display = self.getDevice("display_1")

        # Set the camera properties
        self.camera_width, self.camera_height = (
            self.camera.getWidth(),
            self.camera.getHeight(),
        )
        self.frame_area = self.camera_width * self.camera_height

        # Set the target properties
        self.center_x = self.camera_width / 2
        self.lower_y = self.camera_height + LOWER_Y
        self.lower_center = [self.center_x, self.lower_y]
        self.tolerance_x = 1

        # Set color range for target detection
        color_tolerance = 5
        self.color_target = np.array([46, 52, 54])
        self.lower_color = self.color_target - color_tolerance
        self.upper_color = self.color_target + color_tolerance

        # set the action spaces: 0 = left, 1 = right
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # set the observation space: (channels, camera_height, camera_width)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4,),
            dtype=np.uint16,
        )

        # Set the initial state
        self.state = np.zeros(4, dtype=np.uint16)

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
        self.state = np.zeros(4, dtype=np.uint16)

        # info dictionary
        info: dict = {}

        return self.state, info

    def step(self, action):
        # Set the action for the left and right wheels
        left_wheels_action = action[0] * self.max_wheel_speed
        right_wheels_action = action[1] * self.max_wheel_speed

        # Control the wheels
        self.run_wheels(left_wheels_action, "left")
        self.run_wheels(right_wheels_action, "right")

        # Go to the next step
        super().step(self.timestep)

        # Get the new observation
        self.state, target_distance = self.get_observation(self.camera_width, self.camera_height)

        # Calculate the reward
        reward = 0

        # Reward based on the distance from the target
        norm_target_distance = 1 / (
            1 + (10 ** (4 * target_distance - self.distance_threshold))
        )
        reward_color = norm_target_distance * (10 ** -3)

        # Robot reach target
        reach_target = target_distance <= self.distance_threshold
        reward_reach_target = 10 if reach_target else 0

        # Robot move too far from the initial position
        pos = self.robot.getPosition()
        robot_distance = (
            (pos[0] - self.init_pos[0]) ** 2 + (pos[1] - self.init_pos[1]) ** 2
        ) ** 0.5
        robot_far_away = robot_distance > self.max_robot_distance
        robot_distance_punishment = -5 if robot_far_away else 0

        # Robot hitting the arena boundaries
        hit_arena = not (
            self.arena_x_min <= pos[0] <= self.arena_x_max
            and self.arena_y_min <= pos[1] <= self.arena_y_max
        )
        hit_arena_punishment = -5 if hit_arena else 0

        # Calculate the final reward
        reward = (
            reward_color
            + reward_reach_target
            + robot_distance_punishment
            + hit_arena_punishment
        )

        # Check if the episode is done
        done = reach_target or robot_far_away or hit_arena

        return self.state, reward, done, False, {}

    def render(self, mode: str = "human") -> Any:
        pass

    def get_observation(self, width, height):
        image = self.camera.getImage()

        # Extract RGB channels from the image
        red_channel, green_channel, blue_channel = self.extract_rgb_channels(
            image, width, height
        )
        self.img_rgb = [red_channel, green_channel, blue_channel]

        # Perform the recognition process
        self.state, distance = self.recognition_process(self.img_rgb, width, height)

        return self.state, distance

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

    def recognition_process(self, image, width, height):
        target_px, distance, centroid = 0, None, [None, None]
        target_x_min, target_x_max, target_y_min, target_y_max = width, 0, height, 0

        for y in range(height):
            for x in range(width):
                r, g, b = image[0][y][x], image[1][y][x], image[2][y][x]
                if (
                    self.lower_color[0] <= r <= self.upper_color[0]
                    and self.lower_color[1] <= g <= self.upper_color[1]
                    and self.lower_color[2] <= b <= self.upper_color[2]
                ):
                    target_px += 1
                    target_x_min, target_x_max = min(target_x_min, x), max(
                        target_x_max, x
                    )
                    target_y_min, target_y_max = min(target_y_min, y), max(
                        target_y_max, y
                    )

        # If the target is not detected
        if target_px == 0:
            return np.zeros(4, dtype=np.uint16), None

        # Set the new state
        self.state = np.array(
            [target_x_min, target_x_max, target_y_min, target_y_max], dtype=np.uint16
        )

        # Calculate the centroid and distance from the target
        centroid = [
            (target_x_max + target_x_min) / 2,
            (target_y_max + target_y_min) / 2,
        ]
        distance = np.sqrt(
            (centroid[0] - self.lower_center[0]) ** 2
            + (centroid[1] - self.lower_center[1]) ** 2
        )

        return self.state, distance

    def set_arena_boundaries(self):
        arena_tolerance = 1.0
        size_field = self.floor.getField("floorSize").getSFVec3f()
        x, y = size_field
        self.arena_x_max, self.arena_y_max = (
            x / 2 - arena_tolerance,
            y / 2 - arena_tolerance,
        )
        self.arena_x_min, self.arena_y_min = -self.arena_x_max, -self.arena_y_max

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

    def run_wheels(self, velocity, wheel):
        wheels = self.left_wheels if wheel == "left" else self.right_wheels
        for motor in wheels:
            motor.setVelocity(velocity)


# register the environment
register(
    id=ENV_ID,
    entry_point=lambda: Color_VisuoExcaRobo(),
    max_episode_steps=MAX_EPISODE_STEPS,
)
