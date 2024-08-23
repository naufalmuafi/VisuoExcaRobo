import math
from controller import Supervisor

MAX_MOTOR_SPEED = 0.2

# Create the Robot instance
robot = Supervisor()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# List of names of the motors and sensors
names = ["turret", "arm_connector", "lower_arm", "uppertolow", "scoop"]

# Initialize motors for all wheels
wheel_motors = [robot.getDevice(f"wheel{i}_motor") for i in range(1, 5)]

# Initialize motors and sensors for all joints as dictionaries
motors = {name: robot.getDevice(f"{name}_motor") for name in names}
sensors = {name: robot.getDevice(f"{name}_sensor") for name in names}

# Set motors to velocity control mode
for wheel_motor in wheel_motors:
    wheel_motor.setPosition(float("inf"))
    wheel_motor.setVelocity(0.0)

# Set motors to position and velocity control mode
for motor in motors.values():
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# Enable the sensors
for sensor in sensors.values():
    sensor.enable(timestep)


def run_all_wheels(velocity):
    for wheel_motor in wheel_motors:
        wheel_motor.setVelocity(velocity)


# Function to calculate the shortest rotation direction and angle difference
def calculate_turret_movement(current_angle, target_angle):
    diffA = math.degrees(target_angle) - math.degrees(current_angle)
    diffB = 360 - diffA

    if diffA < diffB or diffA == diffB:
        direction = 1
    elif diffA > diffB:
        direction = -1

    return direction


# Function to move turret to a specific angle in degrees
def move_turret_to_angle(target_angle_degrees, turret_speed=MAX_MOTOR_SPEED):
    target_angle_radians = math.radians(
        target_angle_degrees
    )  # Convert target angle to radians

    # Get the current position of the turret
    current_angle = sensors["turret"].getValue()

    # Calculate the difference to the target position
    direction = calculate_turret_movement(current_angle, target_angle_radians)

    # move the turret to the target position
    motors["turret"].setVelocity(turret_speed * direction)


def move_arm_connector(
    direction, min_position=-1.1, max_position=1.1, velocity=MAX_MOTOR_SPEED
):
    current_position = sensors["arm_connector"].getValue()

    # Check if the motor is within the defined range
    if min_position <= current_position <= max_position:
        if direction == 0:
            motors["arm_connector"].setVelocity(velocity)
        elif direction == 1:
            motors["arm_connector"].setVelocity(-velocity)
    else:
        motors["arm_connector"].setVelocity(0.0)


def move_lower_arm(
    direction, min_position=-0.27, max_position=0.27, velocity=MAX_MOTOR_SPEED
):
    current_position = sensors["lower_arm"].getValue()

    # Check if the motor is within the defined range
    if min_position <= current_position <= max_position:
        if direction == 0:
            motors["lower_arm"].setVelocity(velocity)
        elif direction == 1:
            motors["lower_arm"].setVelocity(-velocity)
    else:
        motors["lower_arm"].setVelocity(0.0)


def move_uppertolow(
    direction, min_position=-0.9, max_position=0.9, velocity=MAX_MOTOR_SPEED
):
    current_position = sensors["uppertolow"].getValue()

    # Check if the motor is within the defined range
    if min_position <= current_position <= max_position:
        if direction == 0:
            motors["uppertolow"].setVelocity(velocity)
        elif direction == 1:
            motors["uppertolow"].setVelocity(-velocity)
    else:
        motors["uppertolow"].setVelocity(0.0)


def move_scoop(
    direction, min_position=-1.1, max_position=1.1, velocity=MAX_MOTOR_SPEED
):
    current_position = sensors["scoop"].getValue()

    # Check if the motor is within the defined range
    if min_position <= current_position <= max_position:
        if direction == 0:
            motors["scoop"].setVelocity(velocity)
        elif direction == 1:
            motors["scoop"].setVelocity(-velocity)
    else:
        motors["scoop"].setVelocity(0.0)


# Main loop:
start_time = robot.getTime()

while robot.step(timestep) != -1:
    # Set velocity for the first 3 seconds, then stop
    duration = robot.getTime() - start_time

    # if duration <= 3.0:
    # motors["arm_connector"].setVelocity(0.2)
    # run_all_wheels(1.0)
    # motors["turret"].setVelocity(0.2)
    # elif duration > 3.0 and duration < 5.0:
    # motors["arm_connector"].setVelocity(0.0)
    # run_all_wheels(0.0)
    # motors["turret"].setVelocity(0.0)
    # elif duration >= 5.0 and duration <= 8.0:
    #     run_all_wheelrs(-1.0)
    # elif duration > 8.0:
    #     run_all_wheels(0.0)

    move_scoop(1)
    print(sensors["scoop"].getValue())
