import VisuoExcaRobo as ver
from argparser import parse_arguments

# main program
if __name__ == "__main__":
    # Get the parsed arguments from parser.py
    args = parse_arguments()

    # Instantiate the VisuoExcaRobo class
    robot = ver.VisuoExcaRobo(args)

    # Check the environment
    ready = robot.check_environment()

    if ready:
        print(f"Environment is ready: {robot.env}")

        # Train the model
        model_file = robot.train_PPO(batch_size=args.timesteps, learning_rate=1e-4)

        print("Training is finished, press `Y` for replay...")
        robot.wait_for_y()

        # Test the environment
        print("Testing the Environment with Predicted Value")
        robot.test_PPO(model_file=model_file)
