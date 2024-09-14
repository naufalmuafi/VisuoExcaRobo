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
            self.test(
                "test_1",
                max_steps=self.timesteps,
                model_dir=str(self.model_path),
            )
        elif self.duty == "test_2":
            self.test(
                "test_2",
                max_steps=self.timesteps,
                model_dir=str(self.model_path),
            )
        elif self.duty == "test_3":
            self.final_test(
                max_steps=self.timesteps,
                model_dir=str(self.model_path),
            )
        else:
            print("Invalid duty. Please specify 'train' or 'test'!")

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

    def test(
        self, test_type: str = "test_1", max_steps: int = 3000, model_dir: str = None
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
        (
            step_list,
            reward_list,
            position_list,
            deviation_x_list,
            deviation_y_list,
            target_area_list,
            centroid_list,
        ) = (
            [],
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
            position, deviation_x, deviation_y, target_area, coordinate = (
                info["positions"],
                info["deviation_x"],
                info["deviation_y"],
                info["target_area"],
                info["coordinates"],
            )

            # Calculate centroid of the target area
            x_min, y_min, x_max, y_max = coordinate
            x_centroid = (x_min + x_max) / 2
            y_centroid = (y_min + y_max) / 2
            centroid = (x_centroid, y_centroid)

            # Print the information
            print(
                reward, done, position, deviation_x, deviation_y, target_area, centroid
            )

            # Append the step and reward to the lists
            step_list.append(step)
            reward_list.append(reward)
            position_list.append(position)
            deviation_x_list.append(deviation_x)
            deviation_y_list.append(deviation_y)
            target_area_list.append(target_area)
            centroid_list.append(centroid)

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

        # print last values of each list
        centroid_init = centroid_list[0]

        print(self.env_type, test_type)
        print(f"reward_list: {reward_list[-1]}")
        print(f"target_area_list: {target_area_list[-1]}")
        print(f"deviation_x_list: {deviation_x_list[-1]}")
        print(f"deviation_y_list: {deviation_y_list[-1]}")

        # Plot the results
        self.plot_results(test_type, reward_list, "Reward", "Plot Reward Over Time")
        self.plot_results(
            test_type, deviation_x_list, "Deviation X", "Plot Deviation X Over Time"
        )
        self.plot_results(
            test_type, deviation_y_list, "Deviation Y", "Plot Deviation Y Over Time"
        )
        self.plot_results(
            test_type, target_area_list, "Target Area", "Plot Target Area Over Time"
        )
        self.excavator_trajectory(test_type, position_list)
        self.centroid_trajectory(test_type, centroid_list, centroid_init)

    def final_test(self, max_steps: int = 3000, model_dir: str = None) -> None:
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
        step_list, reward_list, position_list, distance_list, centroid_list = (
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
            position, distance, coordinate = (
                info["positions"],
                info["distance"],
                info["coordinates"],
            )

            # Calculate centroid of the target area
            x_min, y_min, x_max, y_max = coordinate
            x_centroid = (x_min + x_max) / 2
            y_centroid = (y_min + y_max) / 2
            centroid = (x_centroid, y_centroid)

            # Print the information
            print(reward, done, position, distance, centroid)

            # Append the step and reward to the lists
            step_list.append(step)
            reward_list.append(reward)
            position_list.append(position)
            distance_list.append(distance)
            centroid_list.append(centroid)

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
        centroid_init = centroid_list[0]    
        
        print(self.env_type, "test_3")
        print(f"reward_list: {reward_list[-1]}")    
        print(f"distance_list: {distance_list[-1]}")

        self.plot_results("test_3", reward_list, "Reward", "Plot Reward Over Time")
        self.plot_results(
            "test_3", distance_list, "Distance", "Plot Distance Over Time"
        )
        self.excavator_trajectory("test_3", position_list)
        self.centroid_trajectory("test_3", centroid_list, centroid_init)

    def plot_results(self, test_type, feature, label_name, title) -> None:
        # blueprints of the plot
        output_dir = f"results_{test_type}/{self.env_type}_{self.today_date}/"
        os.makedirs(output_dir, exist_ok=True)

        plt.figure()
        plt.plot(feature, label=label_name)
        plt.xlabel("Time Steps")
        plt.ylabel(label_name)
        plt.title(title)
        plt.savefig(output_dir + f"{label_name}.png")

    def excavator_trajectory(self, test_type, positions):
        output_dir = f"results_{test_type}/{self.env_type}_{self.today_date}/"
        os.makedirs(output_dir, exist_ok=True)

        x_pos, y_pos = zip(*positions)
        plt.figure()
        plt.plot(x_pos, y_pos, color="b", label="Excavator Trajectory Path")
        plt.scatter([-4], [0], color="g", label="Excavator Initial Position")
        plt.scatter([3.5], [-2], color="r", marker="*", s=150, label="Rock Position")
        plt.title(f"Excavator Movement Trajectory")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)
        plt.xlim([-5, 5])
        plt.ylim([-3, 3])
        plt.savefig(output_dir + "excavator_trajectory.png")

    def centroid_trajectory(self, test_type, centroid, init_position):
        output_dir = f"results_{test_type}/{self.env_type}_{self.today_date}/"
        os.makedirs(output_dir, exist_ok=True)

        # Filter out (0, 0) positions from the centroid list
        filtered_centroid = [(x, y) for x, y in centroid if not (x == 0 and y == 0)]
        
        if filtered_centroid:
            x_pos, y_pos = zip(*filtered_centroid)
        else:
            x_pos, y_pos = [], []
        x_init, y_init = init_position
        
        plt.figure()
        plt.plot(x_pos, y_pos, color="b", label="Centroid Trajectory Path", zorder=3)
        plt.scatter(x_init, y_init, color="g", label="Initial Position", zorder=4)
                
        if test_type == "test_3":
            plt.scatter(120, 90, color="r", marker="*", s=150, label="Target Point", zorder=2)                
            plt.scatter(120, 90, color="yellow", alpha=0.5, s=300, label="Goal Area", zorder=1)                
        elif test_type == "test_1" or test_type == "test_2":
            # Create rectangle for the goal area
            rect_width = 129 - 127  # x_max - x_min
            rect_height = 128 - 85.33  # Assuming y_max is the full height (128) since it's not provided
            rectangle = plt.Rectangle((127, 85.33), rect_width, rect_height, color="r", alpha=0.6, zorder=1, label="Goal Area")
            plt.gca().add_patch(rectangle)

        # Invert Y axis so (0,0) is at top-left corner
        plt.xlim([0, 256])
        plt.ylim([128, 0])  # Inverted Y-axis

        plt.title("Centroid Movement Trajectory in Frame")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True)

        plt.savefig(output_dir + "centroid_trajectory.png")

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
