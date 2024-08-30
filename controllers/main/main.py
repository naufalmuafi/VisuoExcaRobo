import VisuoExcaRobo as ver
from argparser import parse_arguments

TIMESTEPS = 1000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

# main program
if __name__ == "__main__":
    # Get the parsed arguments from parser.py
    args = parse_arguments(TIMESTEPS)

    # Instantiate the VisuoExcaRobo class
    robot = ver.VisuoExcaRobo(args)

    # Check the environment
    ready = robot.check_environment()

    if ready:
        print(f"Environment is ready: {robot.env}")

        # Train the model
        model_file = robot.train_PPO(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

        print("Training is finished, press `Y` for replay...")
        robot.wait_for_y()

        # Test the environment
        # model_file = "models/20240829_ppo_1000000_bs_1024_lr_1e-04"
        print("Testing the Environment with Predicted Value")
        robot.test_PPO(model_file=model_file)
