import Color_VisuoExcaRobo
import YOLO_VisuoExcaRobo

import os
import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from typing import Tuple, Callable
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class VisuoExcaRobo:
    """
    A class representing the VisuoExcaRobo environment for training and testing
    a Proximal Policy Optimization (PPO) model.

    Attributes:
        duty (str): The task to perform, either 'train' or 'test'.
        env_type (str): The type of environment ('color' or 'YOLO').
        timesteps (int): The number of timesteps for training.
        model_name (str): The name of the model directory.
        log_name (str): The name of the log directory.
        env_id (str): The identifier for the gym environment.
        env (gym.Env): The gym environment instance.
        model_dir (str): The directory path for saving models.
        log_dir (str): The directory path for saving logs.
        today_date (str): The current date in YYYYMMDD format.
    """

    def __init__(self, args) -> None:
        """
        Initializes the VisuoExcaRobo class with given arguments.

        Args:
            args: The arguments passed to the class, typically a parsed
                  command-line argument object.
        """
        # Extract the arguments
        (
            self.duty,
            self.env_type,
            self.timesteps,
            self.model_name,
            self.log_name,
        ) = self.extract_args(args)

        # Define the environment ID based on the environment type
        if self.env_type == "color":
            self.env_id = "Color_VisuoExcaRobo"
        elif self.env_type == "YOLO":
            self.env_id = "YOLO_VisuoExcaRobo"

        # Create the environment (Single process)
        self.env = gym.make(self.env_id)

        # Uncomment the following lines to create a multiprocess environment
        # num_cpu = 4
        # self.env = DummyVecEnv([self.make_env(self.env_id, i) for i in range(num_cpu)])

        # Set the seed for reproducibility
        self.env.seed(42)

        # Create the directories for model and log storage
        self.model_dir, self.log_dir = self.create_dir(self.model_name, self.log_name)

        # Get the current date in YYYYMMDD format for naming files
        self.today_date = datetime.datetime.now().strftime("%Y%m%d")

    def check_environment(self) -> bool:
        """
        Checks the compatibility of the environment with stable-baselines3.

        Returns:
            bool: True if the environment is valid, False otherwise.
        """
        try:
            print(f"Checking the environment: {self.env}...")
            check_env(self.env)
            return True
        except Exception as e:
            print(f"Environment check failed: {e}")
            print("Please check the environment and try again.")
            return False

    def fit(
        self, batch_size=64, learning_rate=0.0003, model_dir="models/", result_file=""
    ) -> None:
        """
        Fits the model by training or testing it based on the duty specified.

        Args:
            batch_size (int): The size of the batch for training.
            learning_rate (float): The learning rate for the PPO model.
            model_dir (str): The directory where the trained model is saved.
            result_file (str): The filename for saving test results.
        """
        if self.duty == "train":
            self.train_PPO(batch_size=batch_size, learning_rate=learning_rate)
        elif self.duty == "test":
            self.test_PPO(model_dir=model_dir, result_file=result_file)

    def train_PPO(self, batch_size=64, learning_rate=0.0003) -> None:
        """
        Trains the PPO model with the specified parameters.

        Args:
            batch_size (int): The size of the batch for training.
            learning_rate (float): The learning rate for the PPO model.
        """
        # Construct the filenames for model and log storage
        model_filename = f"{self.model_dir}/{self.today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"
        log_filename = f"{self.log_dir}/{self.today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"

        # Instantiate the PPO model with MlpPolicy
        print("Training the model with PPO...")
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=log_filename,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Train the model and save it
        model.learn(total_timesteps=self.timesteps)
        model.save(model_filename)
        print(f"Model saved as {model_filename}")

    def test_PPO(self, model_dir: str = None, result_file: str = None) -> None:
        """
        Tests the trained PPO model and visualizes the rewards over time.

        Args:
            model_dir (str): The directory path where the trained model is stored.
            result_file (str): The filename to save the test results.
        """
        # Load the pre-trained model
        try:
            model = PPO.load(model_dir, env=self.env)
        except FileNotFoundError:
            print(
                "Model not found. Please train the model first/type your model name correctly."
            )
            return

        print("Load Model Successful")
        print("Testing the Environment with Predicted Value")

        # Initialize lists for steps and rewards
        step_list, reward_list = [], []
        step, done = 0, False

        # Reset the environment before testing
        obs, _ = self.env.reset()

        # Run the test loop
        while not done:
            # Predict the action using the model
            action, _states = model.predict(obs)
            obs, reward, done, _, _ = self.env.step(action)

            print(reward, done)
            step_list.append(step)
            reward_list.append(reward)

            # Reset the environment if the episode is done
            if done:
                obs, _ = self.env.reset()

            step += 1

        print("Test Success.")

        # Plot the reward curve
        plt.plot(step_list, reward_list)
        # Save the plot as a PNG file
        plt.savefig(
            f"{self.log_dir}/test_reward_plot_{result_file}_{self.today_date}.png"
        )
        # Display the plot
        plt.show()

    def extract_args(self, args) -> Tuple[str, int, str, str]:
        """
        Extracts arguments from the input argument object.

        Args:
            args: The input arguments, typically from a command-line parser.

        Returns:
            Tuple[str, int, str, str]: Extracted values for duty, env_type, timesteps, model_dir, and log_dir.
        """
        duty = args.duty
        env_type = args.env
        timesteps = args.timesteps
        model_dir = args.model_dir
        log_dir = args.log_dir

        return duty, env_type, timesteps, model_dir, log_dir

    def create_dir(
        self, model_name: str = "models", log_name: str = "logs"
    ) -> Tuple[str, str]:
        """
        Creates directories for saving models and logs.

        Args:
            model_name (str): The directory name for storing models.
            log_name (str): The directory name for storing logs.

        Returns:
            Tuple[str, str]: The paths for the model and log directories.
        """
        # Create directories if they do not exist
        os.makedirs(model_name, exist_ok=True)
        os.makedirs(log_name, exist_ok=True)

        return model_name, log_name

    def make_env(self, env_id: str, rank: int, seed: int = 0) -> Callable:
        """
        Utility function for creating a multiprocessed environment.

        Args:
            env_id (str): The environment ID.
            rank (int): The index of the subprocess.
            seed (int): The initial seed for the random number generator.

        Returns:
            Callable: A callable that initializes the environment.
        """

        def _init() -> gym.Env:
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    def wait_for_y(self) -> None:
        """
        Helper function that waits for user input to continue the process.
        """
        while True:
            user_input = input("Press 'Y' to continue: ")
            if user_input.upper() == "Y":
                print("Continuing process...\n\n")
                break
            else:
                print("Invalid input. Please press 'Y'.")
