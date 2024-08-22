import math
from controller import Supervisor

# Create the Robot instance
robot = Supervisor()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Initialize motors for all wheels
motors = [robot.getDevice(f"wheel{i}_motor") for i in range(1, 5)]

# Initialize turret motor and sensor
turret_motor = robot.getDevice("turret_motor")
turret_sensor = robot.getDevice("turret_sensor")

# Enable the position sensor
turret_sensor.enable(timestep)

# Set motors to velocity control mode
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# Set turret motor to velocity control mode
turret_motor.setPosition(float("inf"))
turret_motor.setVelocity(0.0)


def run_all_motors(velocity):
    for motor in motors:
        motor.setVelocity(velocity)


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
def move_turret_to_angle(target_angle_degrees, turret_speed=0.2):
    target_angle_radians = math.radians(
        target_angle_degrees
    )  # Convert target angle to radians

    # Get the current position of the turret
    current_angle = turret_sensor.getValue()

    # Calculate the difference to the target position
    direction = calculate_turret_movement(current_angle, target_angle_radians)

    # move the turret to the target position
    turret_motor.setVelocity(turret_speed * direction)


# Main loop:
start_time = robot.getTime()

while robot.step(timestep) != -1:
    # Set velocity for the first 3 seconds, then stop
    # duration = robot.getTime() - start_time

    # if duration <= 3.0:
    #     run_all_motors(1.0)
    #     turret_motor.setVelocity(0.2)
    # elif duration > 3.0 and duration < 5.0:
    #     run_all_motors(0.0)
    #     turret_motor.setVelocity(0.0)
    # elif duration >= 5.0 and duration <= 8.0:
    #     run_all_motors(-1.0)
    # elif duration > 8.0:
    #     run_all_motors(0.0)

    move_turret_to_angle(300)
    print(math.degrees(turret_sensor.getValue()))
