import sys
from typing import Any, Tuple, List
from controller import Supervisor, Display

try:
    import math
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

# Constants used in the environment
ENV_ID = "YOLO_VisuoExcaRobo"
MAX_EPISODE_STEPS = 5000
MAX_WHEEL_SPEED = 5.0
MAX_MOTOR_SPEED = 0.7
MAX_ROBOT_DISTANCE = 8.0
LOWER_Y = -20
SHARPNESS = 3
DISTANCE_THRESHOLD = 20


class YOLO_VisuoExcaRobo(Supervisor, Env):
    """
    A custom Gym environment for controlling an excavator robot in Webots using color-based target detection.

    This class integrates the Webots Supervisor with Gymnasium's Env, enabling reinforcement learning tasks.
    """

    def __init__(self, max_episode_steps: int = MAX_EPISODE_STEPS) -> None:
        """
        Initialize the YOLO_VisuoExcaRobo environment.

        Args:
            max_episode_steps (int): The maximum number of steps per episode.
        """
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        random.seed(42)

        # Register the environment with Gym
        self.spec: EnvSpec = EnvSpec(id=ENV_ID, max_episode_steps=max_episode_steps)

        # Get the robot node
        self.robot = self.getFromDef("EXCAVATOR")

        # Set motor and wheel speeds
        self.max_motor_speed = MAX_MOTOR_SPEED
        self.max_wheel_speed = MAX_WHEEL_SPEED
        self.max_robot_distance = MAX_ROBOT_DISTANCE
        self.distance_threshold = DISTANCE_THRESHOLD

        # Get the floor node and set arena boundaries
        self.floor = self.getFromDef("FLOOR")
        self.set_arena_boundaries()

        # Initialize camera and display
        self.camera = self.init_camera()
        self.display = self.getDevice("display_1")

        # Set camera properties
        self.camera_width, self.camera_height = (
            self.camera.getWidth(),
            self.camera.getHeight(),
        )
        self.frame_area = self.camera_width * self.camera_height

        # Target properties
        self.center_x = self.camera_width / 2
        self.lower_y = self.camera_height + LOWER_Y
        self.lower_center = [self.center_x, self.lower_y]
        self.tolerance_x = 1

        # Color range for target detection
        color_tolerance = 5
        self.color_target = np.array([46, 52, 54])
        self.lower_color = self.color_target - color_tolerance
        self.upper_color = self.color_target + color_tolerance

        # Define action space and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4,), dtype=np.uint16
        )

        # Initialize the robot state
        self.state = np.zeros(4, dtype=np.uint16)
        self.seed()

    def reset(self, seed: Any = None, options: Any = None) -> Any:
        """
        Reset the environment to the initial state.

        Args:
            seed (Any): Seed for random number generation.
            options (Any): Additional options for reset.

        Returns:
            Tuple: Initial observation and info dictionary.
        """
        # Reset the simulation
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        # Set robot to initial position
        self.init_pos = self.robot.getPosition()

        # Initialize motors and sensors
        self.wheel_motors, self.motors, self.sensors = self.init_motors_and_sensors()
        self.left_wheels = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels = [self.wheel_motors["rf"], self.wheel_motors["rb"]]

        super().step(self.timestep)

        # Initialize state and return it
        self.state = np.zeros(4, dtype=np.uint16)
        info: dict = {}

        return self.state, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action (np.ndarray): The action to be taken by the robot.

        Returns:
            Tuple: Observation, reward, done flag, truncation flag, and info dictionary.
        """
        # Set the action for left and right wheels
        left_wheels_action = action[0] * self.max_wheel_speed
        right_wheels_action = action[1] * self.max_wheel_speed

        # Control the wheels
        self.run_wheels(left_wheels_action, "left")
        self.run_wheels(right_wheels_action, "right")

        # Proceed to the next simulation step
        super().step(self.timestep)

        # Get new observation and target distance
        self.state, target_distance = self.get_observation(
            self.camera_width, self.camera_height
        )

        # Calculate the reward
        reward_color = self.f(target_distance) * (10**-3)
        reach_target = target_distance <= self.distance_threshold - 10
        reward_reach_target = 10 if reach_target else 0

        # Check robot position relative to its initial position
        pos = self.robot.getPosition()
        robot_distance = (
            (pos[0] - self.init_pos[0]) ** 2 + (pos[1] - self.init_pos[1]) ** 2
        ) ** 0.5
        robot_far_away = robot_distance > self.max_robot_distance
        robot_distance_punishment = -3 if robot_far_away else 0

        # Check if the robot hits the arena boundaries
        arena_th = 1.5
        hit_arena = not (
            self.arena_x_min + arena_th <= pos[0] <= self.arena_x_max - arena_th
            and self.arena_y_min + arena_th <= pos[1] <= self.arena_y_max - arena_th
        )
        hit_arena_punishment = -3 if hit_arena else 0

        # Final reward calculation
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
        """
        Render the environment (not implemented).

        Args:
            mode (str): The mode for rendering.

        Returns:
            Any: Not used.
        """
        pass

    def seed(self, seed=None) -> List[int]:
        """
        Seed the environment for reproducibility.

        Args:
            seed (Any): The seed value.

        Returns:
            List[int]: The list containing the seed used.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def f(
        self,
        target_distance,
        sharpness=SHARPNESS,
        distance_threshold=DISTANCE_THRESHOLD,
    ) -> float:
        """
        Calculate the reward based on the distance to the target using a logistic function.

        Args:
            target_distance (float): Distance from the target.
            sharpness (int): Sharpness factor for the logistic function.
            distance_threshold (float): Threshold distance for target detection.

        Returns:
            float: The reward based on the target distance.
        """
        exponent = sharpness * (target_distance - distance_threshold) * math.log(10)
        try:
            result = 1 / (1 + math.exp(exponent))
        except OverflowError:
            result = 0  # If exponent is too large, the value approaches zero
        return result

    def get_observation(self, width, height) -> Tuple[np.ndarray, float]:
        """
        Capture and process an image from the robot's camera to detect the target.

        Args:
            width (int): Width of the camera frame.
            height (int): Height of the camera frame.

        Returns:
            Tuple: The current state and the distance to the target.
        """
        image = self.camera.getImage()

        # Extract RGB channels from the image
        red_channel, green_channel, blue_channel = self.extract_rgb_channels(
            image, width, height
        )
        self.img_rgb = [red_channel, green_channel, blue_channel]

        # Perform the recognition process
        self.state, distance = self.recognition_process(self.img_rgb, width, height)

        return self.state, distance

    def extract_rgb_channels(
        self, image, width, height
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """
        Extract the RGB channels from the camera image.

        Args:
            image (Any): The image captured by the camera.
            width (int): Width of the camera frame.
            height (int): Height of the camera frame.

        Returns:
            Tuple: Red, Green, and Blue channels as lists of lists.
        """
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

    def recognition_process(self, image, width, height) -> Tuple[np.ndarray, float]:
        """
        Process the image to detect the target object based on color.

        Args:
            image (List[List[int]]): RGB channels of the image.
            width (int): Width of the camera frame.
            height (int): Height of the camera frame.

        Returns:
            Tuple: The state array and distance to the target.
        """
        target_px, distance, centroid = 0, 300, [None, None]
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
            return np.zeros(4, dtype=np.uint16), distance

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

    def set_arena_boundaries(self) -> None:
        """
        Set the boundaries of the arena based on the floor node size.
        """
        arena_tolerance = 1.0
        size_field = self.floor.getField("floorSize").getSFVec3f()
        x, y = size_field
        self.arena_x_max, self.arena_y_max = (
            x / 2 - arena_tolerance,
            y / 2 - arena_tolerance,
        )
        self.arena_x_min, self.arena_y_min = -self.arena_x_max, -self.arena_y_max

    def init_camera(self) -> Any:
        """
        Initialize the camera device and enable recognition.

        Returns:
            Any: The initialized camera device.
        """
        camera = self.getDevice("cabin_camera")
        camera.enable(self.timestep)
        camera.recognitionEnable(self.timestep)
        camera.enableRecognitionSegmentation()
        return camera

    def init_motors_and_sensors(self) -> Tuple[dict, dict, dict]:
        """
        Initialize the motors and sensors of the robot.

        Returns:
            Tuple: Dictionaries of wheel motors, arm motors, and sensors.
        """
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

    def run_wheels(self, velocity, wheel) -> None:
        """
        Set the velocity for the robot's wheels.

        Args:
            velocity (float): Speed to set for the wheels.
            wheel (str): Specifies which wheels to move ('left' or 'right').
        """
        wheels = self.left_wheels if wheel == "left" else self.right_wheels
        for motor in wheels:
            motor.setVelocity(velocity)


# Register the environment
register(
    id=ENV_ID,
    entry_point=lambda: YOLO_VisuoExcaRobo(),
    max_episode_steps=MAX_EPISODE_STEPS,
)
