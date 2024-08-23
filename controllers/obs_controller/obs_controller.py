import math
from controller import Supervisor, Display

MAX_MOTOR_SPEED = 0.7

# Initialize the robot instance and timestep
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# List of names of the motors and sensors
names = ["turret", "arm_connector", "lower_arm", "uppertolow", "scoop"]

# Initialize motors and sensors
wheel_motors = [robot.getDevice(f"wheel{i}_motor") for i in range(1, 5)]
motors = {name: robot.getDevice(f"{name}_motor") for name in names}
sensors = {name: robot.getDevice(f"{name}_sensor") for name in names}

# Configure motor modes
for motor in wheel_motors + list(motors.values()):
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# Enable sensors
for sensor in sensors.values():
    sensor.enable(timestep)

# Initialize camera device
camera = robot.getDevice("cabin_camera")
camera.enable(timestep)
camera.recognitionEnable(timestep)
camera.enableRecognitionSegmentation()

width = camera.getWidth()
height = camera.getHeight()


def run_all_wheels(velocity):
    for motor in wheel_motors:
        motor.setVelocity(velocity)


def calculate_turret_movement(current_angle, target_angle):
    diffA = math.degrees(target_angle) - math.degrees(current_angle)
    diffB = 360 - diffA

    if diffA < diffB or diffA == diffB:
        direction = 1
    elif diffA > diffB:
        direction = -1

    return direction


def move_turret_to_angle(target_angle_degrees, turret_speed=MAX_MOTOR_SPEED):
    current_angle = sensors["turret"].getValue()
    direction = calculate_turret_movement(
        current_angle, math.radians(target_angle_degrees)
    )
    if current_angle != math.radians(target_angle_degrees):
        motors["turret"].setVelocity(turret_speed * direction)
    else:
        motors["turret"].setVelocity(0.0)


def move_joint(name, direction, min_position, max_position, velocity=MAX_MOTOR_SPEED):
    current_position = sensors[name].getValue()
    if min_position <= current_position <= max_position:
        motors[name].setVelocity(velocity * (1 if direction == 0 else -1))
    else:
        motors[name].setVelocity(0.0)


# 0 is left, 1 is right
def move_arm_connector(
    direction,
    min_position=-1.1,
    max_position=1.1,
    velocity=MAX_MOTOR_SPEED,
    toCenter=False,
):
    current_position = sensors["arm_connector"].getValue()

    if toCenter:
        tolerance = 0.001
        if current_position > tolerance or current_position < -tolerance:
            motors["arm_connector"].setVelocity(
                velocity * (1 if current_position < 0 else -1)
            )
        elif current_position < tolerance or current_position > -tolerance:
            motors["arm_connector"].setVelocity(0.0)
    else:
        # Check if the motor is within the defined range
        if min_position <= current_position <= max_position:
            motors["arm_connector"].setVelocity(
                velocity * (1 if direction == 0 else -1)
            )
        else:
            motors["arm_connector"].setVelocity(0.0)


# 0 is down, 1 is up
def move_lower_arm(
    direction, min_position=-0.27, max_position=0.27, velocity=MAX_MOTOR_SPEED
):
    current_position = sensors["lower_arm"].getValue()

    # Check if the motor is within the defined range
    if min_position <= current_position <= max_position:
        motors["lower_arm"].setVelocity(velocity * (1 if direction == 0 else -1))
    else:
        motors["lower_arm"].setVelocity(0.0)


# 0 is down, 1 is up
def move_uppertolow(
    direction, min_position=-0.9, max_position=0.9, velocity=MAX_MOTOR_SPEED
):
    current_position = sensors["uppertolow"].getValue()

    # Check if the motor is within the defined range
    if min_position <= current_position <= max_position:
        motors["uppertolow"].setVelocity(velocity * (1 if direction == 0 else -1))
    else:
        motors["uppertolow"].setVelocity(0.0)


# 0 is inside, 1 is outside
def move_scoop(
    direction, min_position=-1.1, max_position=1.1, velocity=MAX_MOTOR_SPEED + 0.3
):
    current_position = sensors["scoop"].getValue()

    # Check if the motor is within the defined range
    if min_position <= current_position <= max_position:
        motors["scoop"].setVelocity(velocity * (1 if direction == 0 else -1))
    else:
        motors["scoop"].setVelocity(0.0)


def digging_operation():
    initial_positions = {
        "arm_connector": sensors["arm_connector"].getValue(),
        "lower_arm": sensors["lower_arm"].getValue(),
        "uppertolow": sensors["uppertolow"].getValue(),
        "scoop": sensors["scoop"].getValue(),
    }

    targets = {
        "lower_arm": 0.1,
        "uppertolow": 0.45,
        "scoop": 1.0,
    }

    step = 0
    delay_start_time = None

    while True:
        current_positions = {
            "lower_arm": sensors["lower_arm"].getValue(),
            "uppertolow": sensors["uppertolow"].getValue(),
            "scoop": sensors["scoop"].getValue(),
        }

        if step == 0:
            # Move up with adjusted target for uppertolow
            move_arm_connector(1, toCenter=True)
            move_lower_arm(1, min_position=-targets["lower_arm"])
            move_uppertolow(1, min_position=-targets["uppertolow"] + 0.2)
            move_scoop(1, min_position=-targets["scoop"])

            adjusted_targets = {"uppertolow": -targets["uppertolow"] + 0.2}
            if all(
                current_positions[joint] <= adjusted_targets.get(joint, target)
                for joint, target in {**targets, **adjusted_targets}.items()
            ):
                delay_start_time = robot.getTime()
                step = 1

        elif step == 1:
            if robot.getTime() - delay_start_time >= 1.0:
                step = 2

        elif step == 2:
            # Move down
            move_arm_connector(1, toCenter=True)
            move_lower_arm(0, max_position=targets["lower_arm"])
            move_uppertolow(0, max_position=targets["uppertolow"])
            move_scoop(0, max_position=targets["scoop"] - 0.3)

            adjusted_targets = {"scoop": targets["scoop"] - 0.3}
            if all(
                current_positions[joint] >= adjusted_targets.get(joint, target)
                for joint, target in {**targets, **adjusted_targets}.items()
            ):
                delay_start_time = robot.getTime()
                step = 3

        elif step == 3:
            if robot.getTime() - delay_start_time >= 1.0:
                return [
                    initial_positions[joint]
                    for joint in ["arm_connector", "lower_arm", "uppertolow", "scoop"]
                ]

        robot.step(timestep)


# Main loop:
start_time = robot.getTime()

while robot.step(timestep) != -1:
    # Set velocity for the first 3 seconds, then stop
    duration = robot.getTime() - start_time

    # if duration <= 5.0:
    #     move_arm_connector(1)
    # run_all_wheels(1.0)
    # motors["turret"].setVelocity(0.2)
    # elif duration > 5.0:
    #     move_arm_connector(0, toCenter=True)
    # run_all_wheels(0.0)
    # motors["turret"].setVelocity(0.0)
    # elif duration >= 5.0 and duration <= 8.0:
    # run_all_wheels(-1.0)

    # elif duration > 8.0:
    #     run_all_wheels(0.0)

    # move_turret_to_angle(90)
    step_position = digging_operation()
    print(step_position)
    exit(1)
