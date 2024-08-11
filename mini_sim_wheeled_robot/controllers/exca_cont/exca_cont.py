from controller import Robot, Motor, PositionSensor

# Time step of the simulation
TIME_STEP = 32

# Initialize the Robot instance
robot = Robot()

# Retrieve the motors and position sensors
base_motor = robot.getDevice('link2_to_base')
base_motor.setPosition(float('inf'))  # Set to infinite position to allow continuous rotation
base_motor.setVelocity(0.05)  # Set the speed

link3_motor = robot.getDevice('link3_to_link2')
link3_motor.setPosition(float('inf'))
link3_motor.setVelocity(0.1)

link4_motor = robot.getDevice('link4_to_link3')
link4_motor.setPosition(float('inf'))
link4_motor.setVelocity(0.1)

link5_motor = robot.getDevice('link5_to_link4')
link5_motor.setPosition(float('inf'))
link5_motor.setVelocity(0.1)

# Sensors to read the current position of the joints (if needed)
base_sensor = robot.getDevice('link2_to_base_sensor')
base_sensor.enable(TIME_STEP)

link3_sensor = robot.getDevice('link3_to_link2_sensor')
link3_sensor.enable(TIME_STEP)

link4_sensor = robot.getDevice('link4_to_link3_sensor')
link4_sensor.enable(TIME_STEP)

link5_sensor = robot.getDevice('link5_to_link4_sensor')
link5_sensor.enable(TIME_STEP)

# Main control loop
while robot.step(TIME_STEP) != -1:
    # You can read sensor values here if needed
    base_position = base_sensor.getValue()
    link3_position = link3_sensor.getValue()
    link4_position = link4_sensor.getValue()
    link5_position = link5_sensor.getValue()

    # Print the sensor values (optional)
    print(f'Base Position: {base_position}')
    print(f'Link 3 Position: {link3_position}')
    print(f'Link 4 Position: {link4_position}')
    print(f'Link 5 Position: {link5_position}')

    # Set the velocity or position here
    # base_motor.setVelocity(0.1)  # Example of changing velocity dynamically

    # You can also set specific positions if needed
    # base_motor.setPosition(1.0)

    # If you want to stop the robot after a specific time, use robot.getTime()
    # For example, if you want to stop after 10 seconds:
    if robot.getTime() > 10.0:
        base_motor.setVelocity(0)
        link3_motor.setVelocity(0)
        link4_motor.setVelocity(0)
        link5_motor.setVelocity(0)
        break

# You can add more complex behavior, like controlling specific movements,
# checking collisions, or interacting with the environment.
