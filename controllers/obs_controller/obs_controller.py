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


# Function to move turret to a specific angle (in degrees)
def move_turret_to_angle_degrees(angle_degrees):
    # Convert degrees to radians
    angle_radians = math.radians(angle_degrees)
    turret_motor.setPosition(angle_radians)


def run_all_motors(velocity):
    for motor in motors:
        motor.setVelocity(velocity)


def run_turret_to(angle, turret_speed=0.2):
    init_pos = turret_sensor.getValue()
    target_pos = math.radians(angle)

    routeA = angle - math.degrees(init_pos)
    routeB = (2 * math.pi) - routeA
    direction = 1 if routeA < routeB else -1

    while init_pos != target_pos:
        turret_motor.setVelocity(turret_speed * direction)
        init_pos = turret_sensor.getValue()
        print(math.degrees(init_pos))


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

    run_turret_to(90)
