"""

YOLO Target Control
for Excavator Robot

This controller is used to control the excavator robot to find the target object using YOLO control method.

by: Naufal Mu'afi

"""

import cv2
import random
import numpy as np
from ultralytics import YOLO
from controller import Supervisor


MAX_MOTOR_SPEED = 0.7
LOWER_Y = -20
DISTANCE_THRESHOLD = 1.0


class YOLOControl(Supervisor):
    def __init__(self):
        # Initialize the supervisor class
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

        # Initialize the camera, motors, and sensors
        self.camera = self.init_camera()
        self.display = self.getDevice("display_1")
        self.wheel_motors, self.motors, self.sensors = self.init_motors_and_sensors()
        self.left_wheels = [self.wheel_motors["lf"], self.wheel_motors["lb"]]
        self.right_wheels = [self.wheel_motors["rf"], self.wheel_motors["rb"]]

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
        self.yolo_model = YOLO("../../runs/detect/train_m_100/weights/best.pt")

        # Create a window for displaying the processed image
        cv2.namedWindow("Display_2", cv2.WINDOW_AUTOSIZE)

        # Set initial move
        self.initial_move = random.choice([0, 1])

        # Set the initial state
        self.state = np.zeros(4, dtype=np.uint16)

    def run(self):
        while self.step(self.timestep) != -1:
            self.state, distance, centroid = self.get_observation()
            if self.is_done(distance, centroid):
                print("sip.")
                # self.digging_operation()
                exit(1)

            # Wait for a short time (1 ms) to allow the image to be displayed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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

    def get_observation(self):
        # Get the image from the Webots camera (BGRA format)
        img_bgr = self._get_image_in_display()

        # Perform object detection with YOLO
        results = self.yolo_model.predict(img_bgr)
        result = results[0]

        # Post-process the results
        for box in result.boxes:
            label = result.names[box.cls[0].item()]  # Get the label
            cords = box.xyxy[0].tolist()  # Get the coordinates
            cords = [round(x) for x in cords]  # Round the coordinates
            conf = round(box.conf[0].item(), 2)  # Get the confidence

            print(f"Obj. Type: {label}; Coords: {cords}; Prob.: {conf}")

            if label == "rock":
                # Get the coordinates of the bounding box
                x_min, y_min, x_max, y_max = cords

                # Get the new state
                self.state = [x_min, y_min, x_max, y_max]

                # Calculate the centroid and the distance from the lower center
                centroid = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                distance = np.sqrt(
                    (centroid[0] - self.target_coordinate[0]) ** 2
                    + (centroid[1] - self.target_coordinate[1]) ** 2
                )

                print(
                    f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}); Distance: {distance:.2f}"
                )
                self.move_towards_target(centroid, distance)

            else:
                self.search_target()
                self.state, distance, centroid = (
                    np.zeros(4, dtype=np.uint16),
                    None,
                    [None, None],
                )

            print("---")

        return self.state, distance, centroid

    def _get_image_in_display(self):
        # Get the image from the Webots camera (BGRA format)
        video_reader = self.camera.getImage()

        # Convert the raw image data to a NumPy array
        img_np = np.frombuffer(video_reader, dtype=np.uint8).reshape(
            (self.camera_height, self.camera_width, 4)
        )

        # Convert BGRA to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

        # Draw bounding box with label if state is not empty
        if np.any(self.state):
            self.draw_bounding_box(img_bgr, self.state)

        # Display the image in the OpenCV window
        cv2.imshow("Display_2", img_bgr)

        return img_bgr

    def draw_bounding_box(self, img, state):
        x_min, y_min, x_max, y_max = state

        # Draw the bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red box

        # Prepare the label with the confidence score
        label = f"{self.yolo_model.names[state[-1]]} {round(state[-2], 2)}"

        # Get the width and height of the text box
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Draw a filled rectangle for the label background
        cv2.rectangle(img, (x_min, y_min - h - 10), (x_min + w, y_min), (0, 0, 255), -1)

        # Put the label text on the image
        cv2.putText(
            img,
            label,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    def search_target(self):
        # Update display first to ensure UI responsiveness
        self._get_image_in_display()

        print("No target found.")

        if self.initial_move == 0:
            self.run_wheels(self.initial_move, "left")
        elif self.initial_move == 1:
            self.run_wheels(-self.initial_move, "right")

    def move_towards_target(self, centroid, distance):
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
        self.motors["turret"].setVelocity(0.0)
        if direction == "left":
            self.turn_left()
        elif direction == "right":
            self.turn_right()

    def is_done(self, distance, centroid):
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


if __name__ == "__main__":
    controller = YOLOControl()
    controller.run()
