"""

Conventional Color Control
for Excavator Robot

This controller is used to control the excavator robot to find the target object using the conventional color control method.

by: Naufal Mu'afi

"""

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from controller import Supervisor
from typing import Any, Tuple, List, Dict

# Constants for the robot's control
MAX_MOTOR_SPEED: float = 0.7  # Maximum speed for the motors
LOWER_Y: int = -20  # Lower boundary for the y-coordinate
DISTANCE_THRESHOLD: float = (
    1.0  # Distance threshold for considering the target as "reached"
)

# Constants for the testing
MAX_TRIALS: int = 6  # Number of trials to run the testing
MAX_EPISODE_STEPS: int = 2000  # Maximum number of steps per trial

# Create directory for saving plots
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)


class ColorControl(Supervisor):
    """
    ColorControl class controls the excavator robot to find and reach the target using color-based recognition.
    Inherits from Supervisor class of the Webots simulation environment.
    """

    def __init__(self):
        """
        Initializes the robot control system, motors, sensors, and camera for the ColorControl task.
        """
        super().__init__()
        self.timestep: int = int(self.getBasicTimeStep())
        random.seed(42)

        # Get the robot node and setup speed limits
        self.robot = self.getFromDef("EXCAVATOR")
        self.max_motor_speed: float = MAX_MOTOR_SPEED
        self.max_wheel_speed: float = 5.0
        self.distance_threshold: float = DISTANCE_THRESHOLD

        # Set the arena boundaries
        self.floor = self.getFromDef("FLOOR")
        self.set_arena_boundaries()

        # Initialize the camera, motors, and sensors
        self.camera = self.init_camera()

        self.wheel_motors, self.motors, self.sensors = self.init_motors_and_sensors()
        self.left_wheels: List[Any] = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels: List[Any] = [
            self.wheel_motors["rf"],
            self.wheel_motors["rb"],
        ]

        # Camera properties
        self.camera_width, self.camera_height = (
            self.camera.getWidth(),
            self.camera.getHeight(),
        )
        self.frame_area = self.camera_width * self.camera_height

        # Target properties
        self.center_x = self.camera_width / 2
        self.lower_y = self.camera_height + LOWER_Y
        self.target_coordinate = [self.center_x, self.lower_y]
        self.tolerance_x: int = 1

        # Color range for target detection
        color_tolerance: int = 5
        self.color_target: np.ndarray = np.array([46, 52, 54])
        self.lower_color: np.ndarray = self.color_target - color_tolerance
        self.upper_color: np.ndarray = self.color_target + color_tolerance

        # Create a window for displaying processed images
        cv2.namedWindow("Webots Color Recognition Display", cv2.WINDOW_AUTOSIZE)

        # Set initial move direction
        self.initial_move: int = random.choice([0, 1])

        # Initial robot state and results tracking
        self.state: np.ndarray = np.zeros(4, dtype=np.int16)

        # Results tracking
        self.inference_times: Dict[int, List[float]] = {
            i: [] for i in range(MAX_TRIALS)
        }
        self.false_detections: Dict[int, int] = {i: 0 for i in range(MAX_TRIALS)}
        self.trajectory: Dict[int, List[Tuple[float, float]]] = {
            i: [] for i in range(MAX_TRIALS)
        }
        self.coordinates: Dict[int, List[Tuple[float, float]]] = {
            i: [] for i in range(MAX_TRIALS)
        }
        self.init_pos = {i: [] for i in range(MAX_TRIALS)}
        self.success_trials: int = 0
        self.time_to_reach_target: List[float] = []
        self.total_steps: int = 0

    def run(self):
        """
        Main loop of the controller that resets the simulation and continuously
        processes camera input to control the robot.
        """
        self.reset()

        while self.step(self.timestep) != -1:
            self.state, distance, centroid = self.get_observation(
                self.camera_width, self.camera_height
            )
            if self.is_done(distance, centroid):
                print("sip.")
                # self.digging_operation()

                exit(1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def reset(self):
        """
        Resets the simulation environment, reinitializes the robot position, motors, and sensors.
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        # Set the robot to the initial position
        self.init_pos = self.robot.getPosition()

        # Re-initialize motors and sensors after reset
        self.wheel_motors, self.motors, self.sensors = self.init_motors_and_sensors()
        self.left_wheels = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels = [self.wheel_motors["rf"], self.wheel_motors["rb"]]

        super().step(self.timestep)

    def set_arena_boundaries(self):
        """
        Defines the boundaries of the arena based on the size of the floor object in the simulation.
        """
        arena_tolerance: float = 1.0
        size_field = self.floor.getField("floorSize").getSFVec3f()
        x, y = size_field
        self.arena_x_max, self.arena_y_max = (
            x / 2 - arena_tolerance,
            y / 2 - arena_tolerance,
        )
        self.arena_x_min, self.arena_y_min = -self.arena_x_max, -self.arena_y_max

    def init_camera(self) -> Any:
        """
        Initializes the camera and enables recognition for the robot's vision.

        Returns:
            Any: The initialized camera device.
        """
        camera = self.getDevice("cabin_camera")
        camera.enable(self.timestep)
        camera.recognitionEnable(self.timestep)
        return camera

    def init_motors_and_sensors(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Initializes the motors and sensors of the robot for control and movement.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]: Initialized wheel motors, other motors, and sensors.
        """
        # Define motor and sensor names
        names = ["turret", "arm_connector", "lower_arm", "uppertolow", "scoop"]
        wheel = ["lf", "rf", "lb", "rb"]

        # Initialize wheel motors, motors, and sensors
        wheel_motors = {side: self.getDevice(f"wheel_{side}") for side in wheel}
        motors = {name: self.getDevice(f"{name}_motor") for name in names}
        sensors = {name: self.getDevice(f"{name}_sensor") for name in names}

        # Set motor and sensor properties
        for motor in list(wheel_motors.values()) + list(motors.values()):
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
        for sensor in sensors.values():
            sensor.enable(self.timestep)

        return wheel_motors, motors, sensors

    def get_observation(
        self, width: int, height: int
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Captures an image and processes it to extract observation data such as the coordinates of the target,
        distance to the target, and its centroid.

        Args:
            width (int): The width of the camera frame.
            height (int): The height of the camera frame.

        Returns:
            Tuple[np.ndarray, float, List[float]]: The target coordinates, distance to target, and centroid position.
        """
        image = self.camera.getImage()

        # Extract RGB channels from the image
        red_channel, green_channel, blue_channel = self.extract_rgb_channels(
            image, width, height
        )
        self.img_rgb = [red_channel, green_channel, blue_channel]

        # Perform recognition
        coordinate, distance, centroid = self.recognition_process(
            self.img_rgb, width, height
        )

        # Display the image and detected bounding box
        self._get_image_in_display(image, coordinate)

        return coordinate, distance, centroid

    def extract_rgb_channels(
        self, image: Any, width: int, height: int
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """
        Extracts the Red, Green, and Blue channels from the camera image.

        Args:
            image (Any): The raw image captured by the robot camera.
            width (int): The width of the camera frame.
            height (int): The height of the camera frame.

        Returns:
            Tuple[List[List[int]], List[List[int]], List[List[int]]]: The extracted red, green, and blue channels.
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

    def _get_image_in_display(self, img: Any, coordinate: np.ndarray) -> Any:
        """
        Processes the captured image and displays the target detection bounding box.

        Args:
            img (Any): The raw image captured by the camera.
            coordinate (np.ndarray): Coordinates of the detected target.

        Returns:
            Any: The processed BGR image.
        """
        # Convert image to NumPy array and then to BGR
        img_np = np.frombuffer(img, dtype=np.uint8).reshape(
            (self.camera_height, self.camera_width, 4)
        )
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

        # Draw bounding box if a target is detected
        if np.any(coordinate != np.zeros(4, dtype=np.uint16)):
            self.draw_bounding_box(img_bgr, coordinate, "Target")

        # Display the image
        cv2.imshow("Webots Color Recognition Display", img_bgr)
        cv2.waitKey(1)

        # Check for false detections based on bounding box size
        width = coordinate[2] - coordinate[0]
        height = coordinate[3] - coordinate[1]

        if width >= 50 and height >= 50:
            cv2.imwrite(f"false_detection_img.png", img_bgr)

        return img_bgr

    def draw_bounding_box(self, img: np.ndarray, cords: List[int], label: str):
        """
        Draws a bounding box around the detected object and labels it.

        Args:
            img (np.ndarray): The image on which to draw the bounding box.
            cords (List[int]): Coordinates of the bounding box.
            label (str): The label of the detected object.
        """
        bb_x_min, bb_y_min, bb_x_max, bb_y_max = cords

        # Draw the bounding box
        cv2.rectangle(img, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (0, 0, 255), 2)

        # Draw the label on the bounding box
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            img, (bb_x_min, bb_y_min - h - 1), (bb_x_min + w, bb_y_min), (0, 0, 255), -1
        )
        cv2.putText(
            img,
            label,
            (bb_x_min, bb_y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

    def recognition_process(
        self, image: List[List[List[int]]], width: int, height: int
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Processes the image to detect the target object by comparing pixel colors.

        Args:
            image (List[List[List[int]]]): The RGB channels of the image.
            width (int): Width of the image frame.
            height (int): Height of the image frame.

        Returns:
            Tuple[np.ndarray, float, List[float]]: Coordinates of the target, distance to it, and its centroid.
        """
        target_px, distance, centroid = 0, 300, [0, 0]
        target_x_min, target_y_min, target_x_max, target_y_max = width, height, 0, 0

        # Loop through each pixel in the image and compare its color to the target color
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

        # If no target pixels are detected, search for the target
        if target_px == 0:
            self.search_target()
            return np.zeros(4, dtype=np.uint16), 300, [0, 0]

        # Calculate the centroid and distance
        obs = np.array(
            [target_x_min, target_y_min, target_x_max, target_y_max], dtype=np.uint16
        )
        centroid = [
            (target_x_max + target_x_min) / 2,
            (target_y_max + target_y_min) / 2,
        ]
        distance = np.sqrt(
            (centroid[0] - self.target_coordinate[0]) ** 2
            + (centroid[1] - self.target_coordinate[1]) ** 2
        )

        # Move the robot towards the target
        self.move_towards_target(centroid, distance)
        return obs, distance, centroid

    def search_target(self):
        """
        Moves the robot in a random direction when no target is detected.
        """
        print("No target found.")
        if self.initial_move == 0:
            self.run_wheels(self.initial_move, "left")
        elif self.initial_move == 1:
            self.run_wheels(-self.initial_move, "right")

    def move_towards_target(self, centroid: List[float], distance: float):
        """
        Moves the robot towards the detected target based on the target's distance and centroid.

        Args:
            centroid (List[float]): Centroid coordinates of the detected target.
            distance (float): Distance to the detected target.
        """
        if distance >= self.distance_threshold or centroid == [None, None]:
            if centroid[0] <= self.center_x - self.tolerance_x:
                self.adjust_turret_and_wheels("left")
            elif centroid[0] >= self.center_x + self.tolerance_x:
                self.adjust_turret_and_wheels("right")
            else:
                self.motors["turret"].setVelocity(0.0)
                self.run_wheels(self.max_wheel_speed, "all")
        else:
            self.stop_robot()

    def is_done(self, distance: float, centroid: List[float]) -> bool:
        """
        Checks if the robot has reached the target based on the distance and centroid.

        Args:
            distance (float): Distance to the target.
            centroid (List[float]): Centroid coordinates of the detected target.

        Returns:
            bool: True if the robot has reached the target, False otherwise.
        """
        if centroid == [None, None]:
            return False

        x_threshold = [
            self.center_x - self.tolerance_x,
            self.center_x + self.tolerance_x,
        ]
        done_center_x = x_threshold[0] <= centroid[0] <= x_threshold[1]
        done_distance = distance <= self.distance_threshold

        if done_distance and done_center_x:
            self.stop_robot()
            return True
        return False

    def adjust_turret_and_wheels(self, direction):
        self.motors["turret"].setVelocity(0.0)
        if direction == "left":
            self.turn_left()
        elif direction == "right":
            self.turn_right()

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

    def test(self):
        """
        Runs a series of trials to evaluate the robot's performance in detecting and reaching the target.
        """
        for trial in range(MAX_TRIALS):
            print(f"Starting trial {trial + 1}/{MAX_TRIALS}")

            # Reset the simulation for the next trial
            self.reset()

            start_time = time.time()
            step_count = 0
            trial_success = False
            inf_time = []
            false_detection_i = 0
            position = []
            coord = []

            # Simulation loop
            while self.step(self.timestep) != -1 and step_count < MAX_EPISODE_STEPS:
                step_start_time = time.time()
                coordinate, distance, centroid = self.get_observation(
                    self.camera_width, self.camera_height
                )

                # Get the current position of the excavator
                pos = self.robot.getPosition()
                x, y, _ = pos
                position.append((x, y))

                # calculate the centroid
                x_min, y_min, x_max, y_max = coordinate
                centroid = ((x_max + x_min) / 2, (y_max + y_min) / 2)
                coord.append(centroid)

                # Check for false detections based on the target size
                width = coordinate[2] - coordinate[0]
                height = coordinate[3] - coordinate[1]
                if width >= 50 and height >= 50:
                    false_detection_i += 1

                # Check if the robot reached the target
                if self.is_done(distance, centroid):
                    trial_success = True
                    time_taken = time.time() - start_time
                    self.time_to_reach_target.append(time_taken)
                    print(f"Target reached in {time_taken:.2f} seconds.")
                    break

                inference_time = time.time() - step_start_time
                inf_time.append(inference_time * 1000)
                step_count += 1

            # Store results
            if trial_success:
                self.success_trials += 1
            self.inference_times[trial] = inf_time
            self.total_steps += step_count
            self.false_detections[trial] = false_detection_i
            self.trajectory[trial] = position
            self.coordinates[trial] = coord

        # Plot and save results
        self.plot_results()    

    def plot_results(self):
        """
        Generates and saves plots to visualize the robot's performance over all trials.
        """
        # Plot Average Inference Time in ms
        avg_inf_time = np.mean(
            [np.mean(inf_time) for inf_time in self.inference_times.values()]
        )
        print(f"Average Inference Time: {avg_inf_time:.2f} ms")

        # Plot Inference Time Distribution
        for trial_num, inf_time in self.inference_times.items():
            plt.figure()
            plt.plot(inf_time)
            plt.title(f"Inference Time Distribution - Test {trial_num + 1}")
            plt.xlabel("Step")
            plt.ylabel("Time (ms)")
            plt.savefig(
                os.path.join(output_dir, f"inference_time_trial_{trial_num + 1}.png")
            )
            plt.show()

        # Plot False Detection per Trial
        plt.figure()
        plt.plot(
            self.false_detections.keys(),
            self.false_detections.values(),
            marker="o",
            linestyle="--",
            color="r",
        )
        plt.title(f"False Detection per Trial Test")
        plt.xlabel("Num of Trials")
        plt.ylabel("Num of False Detection")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"false_detection_per_trial.png"))
        plt.show()

        # Plot Success Rate
        success_rate = (self.success_trials / MAX_TRIALS) * 100
        print(f"Success rate: {success_rate}%")

        # Plot Time to Reach Target
        plt.figure()
        plt.hist(self.time_to_reach_target, bins=10)
        plt.title("Time to Reach Target")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Number of Trials")
        plt.savefig(os.path.join(output_dir, "time_to_reach_target.png"))
        plt.show()

        # Print average time to reach target
        avg_time_to_reach_target = np.mean(self.time_to_reach_target)
        print(f"Average Time to Reach Target: {avg_time_to_reach_target:.2f} seconds")

        # Print false detection stats
        total_false_detections = sum(self.false_detections.values())
        print(f"Total False Detections: {total_false_detections} in {MAX_TRIALS} Test")

        # Plot The Excavator Trajectory
        for trial_num, trajectory in self.trajectory.items():
            x_positions, y_positions = zip(*trajectory)
            plt.figure()
            plt.plot(
                x_positions, y_positions, color="b", label="Excavator Trajectory Path"
            )
            plt.scatter([-4], [0], color="g", label="Initial Position")
            plt.scatter([3.5], [-2], color="r", marker="*", s=150, label="Target")
            plt.title(f"Excavator Movement Trajectory - Test {trial_num + 1}")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.grid(True)
            plt.xlim([-5, 5])
            plt.ylim([-3, 3])
            plt.savefig(
                os.path.join(
                    output_dir, f"excavator_movement_trial_{trial_num + 1}.png"
                )
            )
            plt.show()        

        for trial_num, coordinates in self.coordinates.items():
            # Filter out (0, 0) positions from the centroid list
            filtered_centroid = [
                (x, y) for x, y in coordinates if not (x == 0 and y == 0)
            ]

            if filtered_centroid:
                x_pos, y_pos = zip(*filtered_centroid)
            else:
                x_pos, y_pos = [], []            

            plt.figure()
            plt.plot(
                x_pos, y_pos, color="b", label="Centroid Trajectory Path", zorder=3
            )
            plt.scatter(230, 49, color="g", label="Initial Position", zorder=4)

            plt.scatter(
                128, 110, color="r", marker="*", s=150, label="Target Point", zorder=2
            )
            plt.scatter(
                128, 110, color="yellow", alpha=0.5, s=300, label="Goal Area", zorder=1
            )

            # Invert Y axis so (0,0) is at top-left corner
            plt.xlim([0, 256])
            plt.ylim([128, 0])  # Inverted Y-axis

            plt.title("Centroid Movement Trajectory in Frame")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(output_dir, f"centroid_movement_trial_{trial_num + 1}.png")
            )
            plt.show()

        # Save results to a text file
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write(f"Average Inference Time: {avg_inf_time:.2f} ms\n")
            f.write(f"Success rate: {success_rate}%\n")
            f.write(f"Total False Detections: {total_false_detections}\n")
            f.write(
                f"Average Time to Reach Target: {avg_time_to_reach_target:.2f} seconds\n"
            )

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
    # Initialize the controller and start the test
    controller = ColorControl()
    controller.run()
