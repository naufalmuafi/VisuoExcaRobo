import argparse


class DutyAction(argparse.Action):
    """
    Custom argparse action to handle 'duty' argument.

    This class ensures that when the 'train' duty is selected, certain arguments
    such as 'timesteps' are required. Similarly, when the 'test' duty is selected,
    the 'model_path' argument is required.
    """

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        """
        Called when the 'duty' argument is parsed. It sets the value and checks
        for additional required arguments if 'train' is selected. If the required
        arguments are not provided, an error is raised.

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
        # If 'duty' is 'test', ensure 'model_path' is provided
        elif values == "test" and not namespace.model_path and not namespace.plot_name:
            parser.error("Argument --model_path and --plot_name required when duty is 'test'.")


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
        choices=["train", "test", 'test_1', 'test_2', 'test_3'],
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
    
    # Add the 'model_path' argument for specifying model file
    parser.add_argument(
        "-mp",
        "--model_path",
        type=str,
        default="models/",
        help="Filename of the model to load",
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
    
    # Add the 'plot_name' argument for specifying plot name
    parser.add_argument(
        "-pn",
        "--plot_name",
        type=str,
        default="test_1",
        help="Filename to save the test results",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate required arguments based on the duty
    if args.duty == "train":
        if args.timesteps is None:
            parser.error("Argument --timesteps is required when duty is 'train'.")
    elif args.duty == "test":
        if args.model_path and args.plot_name is None:
            parser.error("Argument --model_path and --plot_name is required when duty is 'test'.")

    return args
