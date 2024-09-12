import Color_VisuoExcaRobo
import YOLO_VisuoExcaRobo

import cv2
import os
import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from typing import Tuple, Callable
from matplotlib.patches import Circle
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
            self.model_path,
            self.model_name,
            self.log_name,
            self.plot_name,
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
        self.env.unwrapped.seed(123)

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

    def fit(self, batch_size=64, learning_rate=0.0003) -> None:
        """
        Fits the model by training or testing it based on the duty specified.

        Args:
            batch_size (int): The size of the batch for training.
            learning_rate (float): The learning rate for the PPO model.
        """
        if self.duty == "train":
            self.train_PPO(batch_size=batch_size, learning_rate=learning_rate)
        elif self.duty == "test":
            self.test_PPO(
                max_steps=self.timesteps,
                model_dir=str(self.model_path),
                plot_name=str(self.plot_name),
            )
        elif self.duty == "test_1":
            self.test_1(
                max_steps=self.timesteps,
                model_dir=str(self.model_path),                
            )
        elif self.duty == "test_2":
            self.test_2(
                max_steps=self.timesteps,
                model_dir=str(self.model_path),                
            )

    def train_PPO(self, batch_size=64, learning_rate=0.0003) -> None:
        """
        Trains the PPO model with the specified parameters.

        Args:
            batch_size (int): The size of the batch for training.
            learning_rate (float): The learning rate for the PPO model.
        """
        # Construct the filenames for model and log storage
        model_filename = f"{self.model_dir}/{self.env_type}_{self.today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"
        log_filename = f"{self.log_dir}/{self.env_type}_{self.today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"

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

    def test_PPO(
        self, max_steps: int = 3000, model_dir: str = None, plot_name: str = None
    ) -> None:
        """
        Tests the trained PPO model and visualizes the rewards over time.

        Args:
            max_steps (int): The maximum number of steps to run the test.
            model_dir (str): The directory path where the trained model is stored.
            plot_name (str): The filename to save the test results.
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
                print("Test Success.")

            if step >= max_steps:
                print("Max Steps Reached.")
                done = True

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Test Interrupted.")
                done = True

            step += 1

        print("Test Job Done.")

        # Plot the reward curve
        plt.plot(step_list, reward_list)
        # Save the plot as a PNG file
        plt.savefig(
            f"{self.log_dir}/test_reward_plot_{plot_name}_{self.today_date}.png"
        )

    def test_1(
        self, max_steps: int = 3000, model_dir: str = None
    ) -> None:
        """
        Tests the trained PPO model and visualizes the rewards over time.

        Args:
            max_steps (int): The maximum number of steps to run the test.
            model_dir (str): The directory path where the trained model is stored.
            plot_name (str): The filename to save the test results.
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
        step_list, reward_list, position_list, deviation_x_list, deviation_y_list, target_area_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        step, done = 0, False

        # Reset the environment before testing
        obs, _ = self.env.reset()

        # Run the test loop
        while step <= max_steps:
            # Predict the action using the model
            action, _states = model.predict(obs)
            obs, reward, done, _, info = self.env.step(action)

            # Extract the information from the environment
            position, deviation_x, deviation_y, target_area = (
                info["positions"],
                info["deviation_x"],
                info["deviation_y"],
                info["target_area"],
            )

            # Print the information
            print(reward, done, position, deviation_x, deviation_y, target_area)

            # Append the step and reward to the lists
            step_list.append(step)
            reward_list.append(reward)
            position_list.append(position)
            deviation_x_list.append(deviation_x)
            deviation_y_list.append(deviation_y)
            target_area_list.append(target_area)

            # Reset the environment if the episode is done
            if done:
                obs, _ = self.env.reset()
                print("Test Success.")

                break

            if step >= max_steps:
                print("Max Steps Reached.")
                break

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Test Interrupted.")
                break

            step += 1

        print("Test Job Done. Saving Results...")

        # Plot the results
        self.plot_results("test_1", reward_list, "Reward", "Plot Reward Over Time")
        self.plot_results(
            "test_2", deviation_x_list, "Deviation X", "Plot Deviation X Over Time"
        )
        self.plot_results(
            "test_2", deviation_y_list, "Deviation Y", "Plot Deviation Y Over Time"
        )
        self.plot_results(
            "test_2", target_area_list, "Target Area", "Plot Target Area Over Time"
        )
        self.plot_trajectory("test_2", position_list)
    
    def test_2(
        self, max_steps: int = 3000, model_dir: str = None
    ) -> None:
        """
        Tests the trained PPO model and visualizes the rewards over time.

        Args:
            max_steps (int): The maximum number of steps to run the test.
            model_dir (str): The directory path where the trained model is stored.
            plot_name (str): The filename to save the test results.
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
        step_list, reward_list, position_list, distance_list = (
            [],
            [],
            [],
            [],            
        )
        step, done = 0, False

        # Reset the environment before testing
        obs, _ = self.env.reset()

        # Run the test loop
        while step <= max_steps:
            # Predict the action using the model
            action, _states = model.predict(obs)
            obs, reward, done, _, info = self.env.step(action)

            # Extract the information from the environment
            position, distance = (
                info["positions"],
                info["distance"],                
            )

            # Print the information
            print(reward, done, position, distance)

            # Append the step and reward to the lists
            step_list.append(step)
            reward_list.append(reward)
            position_list.append(position)
            distance_list.append(distance)

            # Reset the environment if the episode is done
            if done:
                obs, _ = self.env.reset()
                print("Test Success.")

                break

            if step >= max_steps:
                print("Max Steps Reached.")
                break

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Test Interrupted.")
                break

            step += 1

        print("Test Job Done. Saving Results...")

        # Plot the results
        self.plot_results("test_3", reward_list, "Reward", "Plot Reward Over Time")
        self.plot_results(
            "test_3", distance_list, "Distance", "Plot Distance Over Time"
        )
        self.plot_trajectory("test_3", position_list)

    def plot_results(self, test_type, feature, label_name, title) -> None:
        # blueprints of the plot
        output_dir = f"test_results_{test_type}/{self.env_type}_{self.today_date}/"
        os.makedirs(output_dir, exist_ok=True)

        plt.figure()
        plt.plot(feature, label=label_name)
        plt.xlabel("Time Steps")
        plt.ylabel(label_name)
        plt.title(title)
        plt.savefig(output_dir + f"{label_name}.png")

    def plot_trajectory(self, test_type, positions):
        output_dir = f"test_results_{test_type}/{self.env_type}_{self.today_date}/"
        os.makedirs(output_dir, exist_ok=True)

        x_pos, y_pos = zip(*positions)
        plt.figure()
        plt.plot(x_pos, y_pos, color="b", label="Excavator Trajectory Path")
        plt.scatter([-4], [0], color="g", label="Initial Position")
        plt.scatter([3.5], [-2], color="r", marker="*", s=150, label="Target")
        plt.title(f"Excavator Movement Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.xlim([-5, 5])
        plt.ylim([-3, 3])
        plt.savefig(output_dir + "trajectory.png")

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
        model_path = args.model_path
        model_dir = args.model_dir
        log_dir = args.log_dir
        plot_name = args.plot_name

        return duty, env_type, timesteps, model_path, model_dir, log_dir, plot_name

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
