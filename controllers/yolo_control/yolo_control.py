"""
YOLO Target Control
for Excavator Robot

This controller is used to control the excavator robot to find the target object using the YOLO object detection model.

Author: Naufal Mu'afi
"""

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
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


class YOLOControl(Supervisor):
    """
    The YOLOControl class controls the excavator robot using the YOLO object detection model.
    It processes camera input to find and move towards a specified target object.

    Attributes:
        timestep (int): Simulation timestep duration.
        robot (Node): The robot node in the simulation.
        max_motor_speed (float): Maximum speed for the robot's motors.
        max_wheel_speed (float): Maximum speed for the robot's wheels.
        distance_threshold (float): Distance threshold to determine if the target is reached.
        floor (Node): The floor node in the simulation.
        camera (Camera): The camera device attached to the robot.
        display (Display): The display device for showing camera images.
        camera_width (int): Width of the camera's resolution.
        camera_height (int): Height of the camera's resolution.
        frame_area (int): Total area of the camera frame.
        center_x (float): Center x-coordinate of the camera frame.
        lower_y (float): Adjusted lower y-coordinate for the target.
        target_coordinate (list): Coordinates representing the target location.
        tolerance_x (int): Tolerance for the x-coordinate when aligning with the target.
        yolo_model (YOLO): The YOLO model used for object detection.
        initial_move (int): Initial random move direction (0 for left, 1 for right).
        state (np.ndarray): Current state of the robot's target observation.
    """

    def __init__(self):
        """
        Initializes the YOLOControl class, setting up the simulation environment,
        camera, motors, and YOLO model.
        """
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        random.seed(42)

        # Get the robot node
        self.robot = self.getFromDef("EXCAVATOR")

        # Set the maximum speed for the motors and wheels
        self.max_motor_speed = MAX_MOTOR_SPEED
        self.max_wheel_speed = 5.0
        self.distance_threshold = DISTANCE_THRESHOLD

        # Get the floor node and set the arena boundaries
        self.floor = self.getFromDef("FLOOR")
        self.set_arena_boundaries()

        # Initialize the camera and display
        self.camera = self.init_camera()

        # Set the camera properties
        self.camera_width, self.camera_height = (
            self.camera.getWidth(),
            self.camera.getHeight(),
        )
        self.frame_area = self.camera_width * self.camera_height

        # Set the target properties
        self.center_x = self.camera_width / 2
        self.lower_y = self.camera_height + LOWER_Y
        self.target_coordinate = [self.center_x, self.lower_y]
        self.tolerance_x = 1

        # Load the YOLO model
        self.yolo_model = YOLO("../../yolo_model/yolov8m.pt")
        self.yolo_model = YOLO("../../runs/detect/train_m_300/weights/best.pt")

        # Create a window for displaying the processed image
        cv2.namedWindow("Webots YOLO Display", cv2.WINDOW_AUTOSIZE)

        # Set initial move
        self.initial_move = random.choice([0, 1])

        # Set the initial state
        self.state = np.zeros(4, dtype=np.uint16)
        self.cords, self.label, self.conf = np.zeros(4, dtype=np.uint16), "", 0

        # Results tracking
        self.inference_times: Dict[int, List[float]] = {
            i: [] for i in range(MAX_TRIALS)
        }
        self.confidence_score: Dict[int, int] = {i: 0 for i in range(MAX_TRIALS)}
        self.trajectory: Dict[int, List[Tuple[float, float]]] = {
            i: [] for i in range(MAX_TRIALS)
        }
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
            self.state, distance, centroid, _ = self.get_observation()
            if self.is_done(distance, centroid):
                print("sip.")
                exit(1)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def reset(self):
        """
        Resets the simulation environment, reinitializes the robot position,
        motors, and sensors.
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super().step(self.timestep)

        # Set the robot to the initial position
        self.init_pos = self.robot.getPosition()

        # Initialize the motors and sensors
        self.wheel_motors, self.motors, self.sensors = self.init_motors_and_sensors()
        self.left_wheels = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels = [self.wheel_motors["rf"], self.wheel_motors["rb"]]

        super().step(self.timestep)

        # Initialize the state
        self.state = np.zeros(4, dtype=np.uint16)

    def set_arena_boundaries(self):
        """
        Sets the boundaries of the arena based on the floor node size, with a tolerance.
        """
        arena_tolerance = 1.0
        size_field = self.floor.getField("floorSize").getSFVec3f()
        x, y = size_field
        self.arena_x_max, self.arena_y_max = (
            x / 2 - arena_tolerance,
            y / 2 - arena_tolerance,
        )
        self.arena_x_min, self.arena_y_min = -self.arena_x_max, -self.arena_y_max

    def init_camera(self):
        """
        Initializes the camera device and enables it for capturing images.

        Returns:
            Camera: The initialized camera device.
        """
        camera = self.getDevice("cabin_camera")
        camera.enable(self.timestep)

        return camera

    def init_motors_and_sensors(self):
        """
        Initializes the motors and sensors of the robot and enables them.

        Returns:
            Tuple[dict, dict, dict]: Dictionaries of wheel motors, arm motors, and sensors.
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

    def get_observation(self):
        """
        Captures an image from the camera, performs object detection using YOLO,
        and processes the results to determine the target's position.

        Returns:
            Tuple[np.ndarray, float, list]: The state array, distance to the target, and centroid of the target.
        """
        # Get the image from the Webots camera (BGRA format)
        img_bgr = self._get_image_in_display()

        distance, centroid, inference_time = 300, [0, 0], 0.0
        x_min, y_min, x_max, y_max = 0, 0, 0, 0
        obs = np.zeros(4, dtype=np.uint16)
        self.cords, self.label, self.conf = np.zeros(4, dtype=np.uint16), "", 0

        # Perform object detection with YOLO
        results = self.yolo_model.predict(img_bgr)
        result = results[0]

        # Get Inferece Time (ms)
        inference_time = result.speed["inference"]

        # Post-process the results (shows only if the object is a rock)
        if result.boxes:
            for box in result.boxes:
                self.label = result.names[box.cls[0].item()]  # Get the label
                self.cords = box.xyxy[0].tolist()  # Get the coordinates
                self.cords = [round(x) for x in self.cords]  # Round the coordinates
                self.conf = round(box.conf[0].item(), 2)  # Get the confidence

                print(
                    f"Obj. Type: {self.label}; Coords: {self.cords}; Prob.: {self.conf}; Inference Time: {inference_time:.2f} ms"
                )

                if self.label == "rock":
                    # Get the coordinates of the bounding box
                    x_min, y_min, x_max, y_max = self.cords

                    # Get the new state
                    obs = [x_min, y_min, x_max, y_max]

                    # Calculate the centroid and the distance from the lower center
                    centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                    distance = np.sqrt(
                        (centroid[0] - self.target_coordinate[0]) ** 2
                        + (centroid[1] - self.target_coordinate[1]) ** 2
                    )

                    print(
                        f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}); Distance: {distance:.2f}"
                    )
                    print(f"Target Area: {(x_max - x_min) * (y_max - y_min)}")
                    self.move_towards_target(centroid, distance)
        else:
            self.search_target()

        test_param = [inference_time, self.conf]
        print(self.cords)
        print("---")

        return obs, distance, centroid, test_param

    def _get_image_in_display(self):
        """
        Captures an image from the Webots camera and processes it for object detection.

        Returns:
            np.ndarray: The processed BGR image.
        """
        # Get the image from the Webots camera (BGRA format)
        video_reader = self.camera.getImage()

        # Convert the raw image data to a NumPy array
        img_np = np.frombuffer(video_reader, dtype=np.uint8).reshape(
            (self.camera_height, self.camera_width, 4)
        )

        # Convert BGRA to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

        # Draw bounding box with label if state is not empty
        if np.any(self.cords):
            self.draw_bounding_box(img_bgr, self.cords, self.label)

        # Display the image in the OpenCV window
        cv2.imshow("Webots YOLO Display", img_bgr)

        return img_bgr

    def draw_bounding_box(self, img, cords, label):
        """
        Draws a bounding box around the detected object and labels it.

        Args:
            img (np.ndarray): The image on which to draw the bounding box.
            cords (list): Coordinates of the bounding box.
            label (str): The label of the detected object.
        """
        bb_x_min, bb_y_min, bb_x_max, bb_y_max = cords

        # Draw the bounding box
        cv2.rectangle(
            img, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (0, 0, 255), 2
        )  # Red box

        # Get the width and height of the text box
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle for the label background
        cv2.rectangle(
            img, (bb_x_min, bb_y_min - h - 1), (bb_x_min + w, bb_y_min), (0, 0, 255), -1
        )

        # Put the label text on the image
        cv2.putText(
            img,
            label,
            (bb_x_min, bb_y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )

    def search_target(self):
        """
        Performs a search pattern when the target object is not detected.
        """
        # Update display first to ensure UI responsiveness
        self._get_image_in_display()

        print("No target found.")

        if self.initial_move == 0:
            self.run_wheels(self.initial_move, "left")
        elif self.initial_move == 1:
            self.run_wheels(-self.initial_move, "right")

    def move_towards_target(self, centroid, distance):
        """
        Moves the robot towards the detected target based on the centroid position
        and distance to the target.

        Args:
            centroid (list): The x and y coordinates of the target's centroid.
            distance (float): The distance from the robot to the target.
        """
        # Update display first to ensure UI responsiveness
        self._get_image_in_display()

        if (distance >= self.distance_threshold) or (centroid == [None, None]):
            if centroid[0] <= self.center_x - self.tolerance_x:
                self.adjust_turret_and_wheels(direction="left")
                print("Adjusting turret and wheels to the left.")
            elif centroid[0] >= self.center_x + self.tolerance_x:
                self.adjust_turret_and_wheels(direction="right")
                print("Adjusting turret and wheels to the right.")
            else:
                self.motors["turret"].setVelocity(0.0)
                self.run_wheels(self.max_wheel_speed, "all")
                print("Moving forward.")
        else:
            self.stop_robot()

    def adjust_turret_and_wheels(self, direction):
        """
        Adjusts the turret and wheels to align the robot with the target.

        Args:
            direction (str): The direction to adjust ('left' or 'right').
        """
        self.motors["turret"].setVelocity(0.0)
        if direction == "left":
            self.turn_left()
        elif direction == "right":
            self.turn_right()

    def is_done(self, distance, centroid):
        """
        Checks if the robot has successfully reached the target.

        Args:
            distance (float): The distance from the robot to the target.
            centroid (list): The x and y coordinates of the target's centroid.

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
            print(
                f"Target achieved. Distance: {distance}; Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})"
            )
            self.stop_robot()
            return True

        return False

    def run_wheels(self, velocity, wheel="all"):
        """
        Sets the velocity for the robot's wheels.

        Args:
            velocity (float): The speed to set for the wheels.
            wheel (str): Specifies which wheels to move ('all', 'left', or 'right').
        """
        wheels = (
            self.left_wheels + self.right_wheels
            if wheel == "all"
            else self.left_wheels if wheel == "left" else self.right_wheels
        )
        for motor in wheels:
            motor.setVelocity(velocity)

    def turn_left(self):
        """
        Turns the robot left by running the left wheels backward and the right wheels forward.
        """
        self.run_wheels(-self.max_wheel_speed, "left")
        self.run_wheels(self.max_wheel_speed, "right")

    def turn_right(self):
        """
        Turns the robot right by running the left wheels forward and the right wheels backward.
        """
        self.run_wheels(self.max_wheel_speed, "left")
        self.run_wheels(-self.max_wheel_speed, "right")

    def stop_robot(self):
        """
        Stops the robot by setting all wheels' velocity to zero.
        """
        self.run_wheels(0.0, "all")


if __name__ == "__main__":
    """
    Main entry point of the YOLOControl script.
    Initializes the YOLOControl instance and starts the control loop.
    """
    controller = YOLOControl()
    controller.run()
