import VisuoExcaRobo as ver
from argparser import parse_arguments

TIMESTEPS = 1000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

# main program
if __name__ == "__main__":
    # Define the temp directories
    __modeldir = "models/"

    # Get the parsed arguments from parser.py
    args = parse_arguments(TIMESTEPS)

    # Instantiate the VisuoExcaRobo class
    robot = ver.VisuoExcaRobo(args)

    # Check the environment
    ready = robot.check_environment()

    if ready:
        print(f"Environment is ready: {robot.env}")

        # Perform the robot
        result_file = "20240901_ppo_5000000_bs_1024_lr_1e-04"
        model_train_dir = __modeldir + result_file

        robot.fit(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            model_dir=model_train_dir,
            result_file=result_file,
        )
