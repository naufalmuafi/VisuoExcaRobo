import argparse


def parse_arguments():
    TIMESTEPS = 1000
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-4

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Train and test the model with 2 options for environment: Color or Object"
    )

    # Add the arguments
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="Color",
        help="Choose the environment to train and test the model: Color or Object",
        choices=["Color", "Object"],
        required=True,
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=TIMESTEPS,
        help="Number of timesteps to train the model",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        default="models",
        help="Directory to store the trained model",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to store the logs",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args
