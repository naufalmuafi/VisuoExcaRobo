import VisuoExcaRobo as ver
from argparser import parse_arguments

# Constants for model training
TIMESTEPS = 1000  # Number of timesteps for training
BATCH_SIZE = 1024  # Batch size for training
LEARNING_RATE = 1e-3  # Learning rate for the PPO model

# Main program entry point
if __name__ == "__main__":
    """
    This is the main entry point of the program where the VisuoExcaRobo model is
    initialized, the environment is checked, and the training or testing is performed
    based on the provided arguments.
    """    

    # Parse the command-line arguments
    args = parse_arguments(TIMESTEPS)

    # Instantiate the VisuoExcaRobo class with parsed arguments
    robot = ver.VisuoExcaRobo(args)

    # Check if the environment is properly set up and ready
    ready = robot.check_environment()

    # If the environment is ready, proceed with training or testing
    if ready:
        print(f"Environment is ready: {robot.env}")        

        # Fit the model by either training or testing it based on 'duty' argument
        robot.fit(
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE            
        )
