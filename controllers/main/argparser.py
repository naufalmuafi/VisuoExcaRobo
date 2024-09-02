import argparse


def parse_arguments(timesteps) -> argparse.Namespace:
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Train or Test the model with 2 options for environment: Color or YOLO"
    )

    # Add the arguments
    parser.add_argument(
        "-d",
        "--duty",
        type=str,
        default="test",
        help="Choose the Duty: train or test",
        choices=["train", "test"],
        required=True,
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="color",
        help="Choose the environment to train and test the model: color or YOLO",
        choices=["color", "YOLO"],
        required=True,
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=timesteps,
        help="Number of timesteps to train the model",
        required=True,
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
