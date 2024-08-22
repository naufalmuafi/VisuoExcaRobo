from controller import Supervisor

# Create the Robot instance
robot = Supervisor()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Initialize motors for all wheels
motors = [robot.getDevice(f"wheel{i}_motor") for i in range(1, 5)]

# Initialize turret motor
turret_motor = robot.getDevice("turret_motor")

# Set motors to velocity control mode
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

# Set turret motor to velocity control mode
turret_motor.setPosition(float("inf"))
turret_motor.setVelocity(0.0)

# Main loop:
start_time = robot.getTime()

while robot.step(timestep) != -1:
    # Set velocity for the first 3 seconds, then stop
    velocity = 1.0 if robot.getTime() - start_time <= 3.0 else 0.0
    for motor in motors:
        motor.setVelocity(velocity)

    # Set turret motor velocity
    turret_motor.setVelocity(1.0)
