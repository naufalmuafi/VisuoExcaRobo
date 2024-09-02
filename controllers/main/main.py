import VisuoExcaRobo as ver
from argparser import parse_arguments

# Constants for model training
TIMESTEPS = 1000  # Number of timesteps for training
BATCH_SIZE = 1024  # Batch size for training
LEARNING_RATE = 1e-4  # Learning rate for the PPO model

# Main program entry point
if __name__ == "__main__":
    """
    This is the main entry point of the program where the VisuoExcaRobo model is
    initialized, the environment is checked, and the training or testing is performed
    based on the provided arguments.
    """

    # Define the model directory where trained models will be stored
    __modeldir = "models/"

    # Parse the command-line arguments
    args = parse_arguments(TIMESTEPS)

    # Instantiate the VisuoExcaRobo class with parsed arguments
    robot = ver.VisuoExcaRobo(args)

    # Check if the environment is properly set up and ready
    ready = robot.check_environment()

    # If the environment is ready, proceed with training or testing
    if ready:
        print(f"Environment is ready: {robot.env}")

        # Define the result file name based on current configurations
        result_file = "20240901_ppo_5000000_bs_1024_lr_1e-04"
        model_train_dir = __modeldir + result_file

        # Fit the model by either training or testing it based on 'duty' argument
        robot.fit(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            model_dir=model_train_dir,
            result_file=result_file,
        )
