import cv2
import numpy as np
from controller import Supervisor

# Initialize the Webots Supervisor and Camera
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
camera = robot.getDevice("cabin_camera")
camera.enable(timestep)

# Create a window for displaying the processed image
cv2.namedWindow("Webots OpenCV Display", cv2.WINDOW_AUTOSIZE)

# Main loop
while robot.step(timestep) != -1:
    # Get the image from the Webots camera (BGRA format)
    video_reader = camera.getImage()

    # Convert the raw image data to a NumPy array
    width = camera.getWidth()
    height = camera.getHeight()
    img_np = np.frombuffer(video_reader, dtype=np.uint8).reshape((height, width, 4))

    # Convert BGRA to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)

    # Add some OpenCV overlays (e.g., draw a rectangle and some text)
    # cv2.rectangle(img_bgr, (50, 50), (300, 300), (0, 255, 0), 3)  # Green rectangle
    # cv2.putText(
    #     img_bgr,
    #     "Webots OpenCV Overlay",
    #     (50, 40),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 0, 255),
    #     2,
    # )  # Red text

    # Display the image in the OpenCV window
    cv2.imshow("Webots OpenCV Display", img_bgr)

    # Wait for a short time (1 ms) to allow the image to be displayed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup and close the OpenCV window
cv2.destroyAllWindows()
