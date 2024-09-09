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
output_dir = "test_results"
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
        self.display = self.getDevice("display_1")
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
        self.inference_times: Dict[int, List[float]] = {
            i: [] for i in range(MAX_TRIALS)
        }
        self.false_detections: Dict[int, int] = {i: 0 for i in range(MAX_TRIALS)}
        self.trajectory: Dict[int, List[Tuple[float, float]]] = {
            i: [] for i in range(MAX_TRIALS)
        }
        self.success_trials: int = 0
        self.time_to_reach_target: List[float] = []
        self.total_steps: int = 0

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

    def test(self):
        """
        Runs multiple trials where the robot attempts to reach the target.
        Tracks performance such as inference time, false detections, and success rate.
        """
        for trial in range(MAX_TRIALS):
            print(f"Starting trial {trial + 1}/{MAX_TRIALS}")
            start_time = time.time()
            step_count = 0
            inf_time, position = [], []

            while self.step(self.timestep) != -1 and step_count < MAX_EPISODE_STEPS:
                # Get observation and robot position
                step_start_time = time.time()
                coordinate, distance, centroid = self.get_observation(
                    self.camera_width, self.camera_height
                )
                x, y, _ = self.robot.getPosition()
                position.append((x, y))

                # Check for false detection based on target size
                width, height = (
                    coordinate[2] - coordinate[0],
                    coordinate[3] - coordinate[1],
                )
                if width >= 50 and height >= 50:
                    self.false_detections[trial] += 1

                # Check if the target has been reached
                if self.is_done(distance, centroid):
                    self.success_trials += 1
                    self.time_to_reach_target.append(time.time() - start_time)
                    print(
                        f"Target reached in {self.time_to_reach_target[-1]:.2f} seconds."
                    )
                    break

                inf_time.append((time.time() - step_start_time) * 1000)
                step_count += 1

            # Store inference times and position trajectory
            self.inference_times[trial] = inf_time
            self.trajectory[trial] = position
            self.total_steps += step_count

            self.reset()

        self.plot_results()

    def plot_results(self):
        """
        Generates plots and prints statistics for the performance metrics across all trials.
        """
        # Calculate and display average inference time
        avg_inf_time = np.mean(
            [np.mean(times) for times in self.inference_times.values()]
        )
        print(f"Average Inference Time: {avg_inf_time:.2f} ms")

        # Plot inference time distribution for each trial
        for trial_num, inf_time in self.inference_times.items():
            plt.plot(inf_time)
            plt.title(f"Inference Time Distribution - Trial {trial_num + 1}")
            plt.xlabel("Step")
            plt.ylabel("Time (ms)")
            plt.savefig(
                os.path.join(output_dir, f"inference_time_trial_{trial_num + 1}.png")
            )
            plt.show()

        # Plot false detection per trial
        plt.plot(
            self.false_detections.keys(),
            self.false_detections.values(),
            marker="o",
            linestyle="--",
            color="r",
        )
        plt.title("False Detection per Trial")
        plt.xlabel("Trial")
        plt.ylabel("False Detections")
        plt.savefig(os.path.join(output_dir, "false_detection_per_trial.png"))
        plt.show()

        # Calculate and display success rate
        success_rate = (self.success_trials / MAX_TRIALS) * 100
        print(f"Success rate: {success_rate}%")

        # Plot time to reach the target for all trials
        plt.hist(self.time_to_reach_target, bins=10)
        plt.title("Time to Reach Target")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Number of Trials")
        plt.savefig(os.path.join(output_dir, "time_to_reach_target.png"))
        plt.show()

        # Display and save additional statistics
        avg_time_to_reach = np.mean(self.time_to_reach_target)
        total_false_detections = sum(self.false_detections.values())
        print(f"Average Time to Reach Target: {avg_time_to_reach:.2f} seconds")
        print(f"Total False Detections: {total_false_detections}")

        # Save key statistics to a file
        with open(os.path.join(output_dir, "results.txt"), "w") as f:
            f.write(f"Average Inference Time: {avg_inf_time:.2f} ms\n")
            f.write(f"Success rate: {success_rate}%\n")
            f.write(f"Total False Detections: {total_false_detections}\n")
            f.write(f"Average Time to Reach Target: {avg_time_to_reach:.2f} seconds\n")


if __name__ == "__main__":
    # Initialize the controller and start the test
    controller = ColorControl()
    controller.test()
