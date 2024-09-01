import cv2
import numpy as np
from controller import Supervisor

# Initialize the Webots Supervisor and Camera
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
camera = robot.getDevice("cabin_camera")
camera.enable(timestep)

# List of names of the motors and sensors
names = ["turret", "arm_connector", "lower_arm", "uppertolow", "scoop"]
wheel = ["lf", "rf", "lb", "rb"]

# Initialize motors and sensors
wheel_motors = {side: robot.getDevice(f"wheel_{side}") for side in wheel}
motors = {name: robot.getDevice(f"{name}_motor") for name in names}
sensors = {name: robot.getDevice(f"{name}_sensor") for name in names}

# Configure motor modes
for motor in list(wheel_motors.values()) + list(motors.values()):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# Enable sensors
for sensor in sensors.values():
    sensor.enable(timestep)

# Create a window for displaying the processed image
cv2.namedWindow("Webots OpenCV Display", cv2.WINDOW_AUTOSIZE)

def run_all_wheels(velocity):
    for motor in wheel_motors.values():
        motor.setVelocity(velocity)

# Main loop
while robot.step(timestep) != -1:
    run_all_wheels(1.0)
    
    # Get the image from the Webots camera (BGRA format)
    video_reader = camera.getImage()

    # Convert the raw image data to a NumPy array
    width = camera.getWidth()
    height = camera.getHeight()
    img_np = np.frombuffer(video_reader, dtype=np.uint8).reshape((height, width, 4))

    # Convert BGRA to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)   

    # Display the image in the OpenCV window
    cv2.imshow("Webots OpenCV Display", img_bgr)

    # Wait for a short time (1 ms) to allow the image to be displayed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup and close the OpenCV window
cv2.destroyAllWindows()
