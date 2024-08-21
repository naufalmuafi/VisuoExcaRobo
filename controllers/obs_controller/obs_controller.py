from controller import Supervisor

# Create the Robot instance
robot = Supervisor()

# Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Initialize motors for all wheels
motors = []

for i in range(1, 5):
    motor = robot.getDevice("wheel" + str(i) + "_motor")
    motor.setVelocity(0.0)
    motor.setPosition(float("inf"))
    motors.append(motor)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    for i in range(4):
        motors[i].setVelocity(1.0)
