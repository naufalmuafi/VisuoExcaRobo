import argparse


class DutyAction(argparse.Action):
    """
    Custom argparse action to handle 'duty' argument.

    This class ensures that when the 'train' duty is selected, certain arguments
    such as 'timesteps' are required.
    """

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        """
        Called when the 'duty' argument is parsed. It sets the value and checks
        for additional required arguments if 'train' is selected.

        Args:
            parser (argparse.ArgumentParser): The argument parser instance.
            namespace (argparse.Namespace): The namespace to hold argument values.
            values (str): The value of the 'duty' argument.
            option_string (str, optional): The option string used in the command line.
        """
        # Set the 'duty' attribute in the namespace
        setattr(namespace, self.dest, values)

        # If 'duty' is 'train', ensure 'timesteps' is provided
        if values == "train" and not namespace.timesteps:
            parser.error("Argument --timesteps required when duty is 'train'.")


def parse_arguments(timesteps: int) -> argparse.Namespace:
    """
    Parses command-line arguments for the program.

    Args:
        timesteps (int): The default number of timesteps for training the model.

    Returns:
        argparse.Namespace: The parsed arguments as an argparse namespace object.
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Train or Test the model with 2 options for environment: Color or YOLO"
    )

    # Add the 'duty' argument with custom action
    parser.add_argument(
        "-d",
        "--duty",
        type=str,
        default="test",
        help="Choose the Duty: train or test",
        choices=["train", "test"],
        required=True,
        action=DutyAction,
    )

    # Add the 'env' argument for environment selection
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="color",
        help="Choose the environment to train and test the model: color or YOLO",
        choices=["color", "YOLO"],
        required=True,
    )

    # Add the 'timesteps' argument for specifying training duration
    parser.add_argument(
        "-t",
        "--timesteps",
        type=int,
        default=timesteps,
        help="Number of timesteps to train the model",
    )

    # Add the 'model_dir' argument for specifying model directory
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        default="models",
        help="Directory to store the trained model",
    )

    # Add the 'log_dir' argument for specifying log directory
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to store the logs",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate required arguments based on the duty
    if args.duty == "train":
        if args.timesteps is None:
            parser.error("Argument --timesteps is required when duty is 'train'.")

    return args
