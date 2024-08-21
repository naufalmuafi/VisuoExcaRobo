from controller import Supervisor

th = 0.05

# Create Supervisor instance
supervisor = Supervisor()

# Get the floor node using DEF name
floor = supervisor.getFromDef('FLOOR')
robot = supervisor.getFromDef("ROBOT")
pos = robot.getPosition()

# Access the size field of the floor
size_field = floor.getField('floorSize').getSFVec3f()
# Print the floor size
print("Floor size:", size_field)

x, y = size_field
x_max, y_max = x/2-th, y/2-th
x_min, y_min = -x_max, -y_max

print(x_max, y_max, x_min, y_min)
print(f"robot: {pos}")