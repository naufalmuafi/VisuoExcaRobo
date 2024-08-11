from controller import Supervisor

TIME_STEP = 32

robot = Supervisor()  # create Supervisor instance

robot_node = robot.getFromDef("ROBOT")  # get the ROBOT node
init_position = robot_node.getPosition()  # get the position of the ROBOT node

# print the initial position of the ROBOT node
print("Initial Position of the Robot: ", init_position)

motors = []
for name in ["left wheel motor", "right wheel motor"]:
    motor = robot.getDevice(name)
    motors.append(motor)
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)

while robot.step(TIME_STEP) != -1:
    motors[0].setVelocity(1.0)
    motors[1].setVelocity(1.0)

    # get the position of the ROBOT node
    pos = robot_node.getPosition()

    print("Current Position of the Robot: ", pos)

    # calculate the distance between the initial and current position
    distance = (
        (pos[0] - init_position[0]) ** 2 + (pos[1] - init_position[1]) ** 2
    ) ** 0.5

    print("Distance: ", distance)

    if distance >= 0.55:
        # stop the robot
        motors[0].setVelocity(0.0)
        motors[1].setVelocity(0.0)
        break
