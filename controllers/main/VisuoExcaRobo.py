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
    def __init__(self, args) -> None:
        self.duty, self.env_type, self.timesteps, self.model_name, self.log_name = (
            self.extract_args(args)
        )

        # Define the environment
        if self.env_type == "color":
            self.env_id = "Color_VisuoExcaRobo"
        elif self.env_type == "YOLO":
            self.env_id = "YOLO_VisuoExcaRobo"

        # Create the environment (Single process)
        self.env = gym.make(self.env_id)

        # Create the environment (Multiple process)
        # num_cpu = 4
        # self.env = DummyVecEnv([self.make_env(self.env_id, i) for i in range(num_cpu)])

        self.env.seed(42)

        # Create the directories
        self.model_dir, self.log_dir = self.create_dir(self.model_name, self.log_name)

        # Get the current date in YYYYMMDD format
        self.today_date = datetime.datetime.now().strftime("%Y%m%d")

    def check_environment(self) -> None:
        # check the environment
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
    ):
        if self.duty == "train":
            self.train_PPO(batch_size=batch_size, learning_rate=learning_rate)
        if self.duty == "test":
            self.test_PPO(model_dir=model_dir, result_file=result_file)

    def train_PPO(self, batch_size=64, learning_rate=0.0003) -> None:
        # Construct the model filename
        model_filename = f"{self.model_dir}/{self.today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"
        log_filename = f"{self.log_dir}/{self.today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"

        # use Proximal Policy Optimization (PPO) algorithm
        print("Training the model with PPO...")
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=log_filename,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Train and save the model
        model.learn(total_timesteps=self.timesteps)
        model.save(model_filename)
        print(f"Model saved as {model_filename}")

    def test_PPO(
        self, steps: int = 500, model_dir: str = None, result_file: str = None
    ) -> None:
        # load the model
        try:
            model = PPO.load(model_dir, env=self.env)
        except FileNotFoundError:
            print(
                "Model not found. Please train the model first/type your model name correctly."
            )
            return
        print("Load Model Successful")
        print("Testing the Environment with Predicted Value")

        # initialize the lists and variables
        step_list, reward_list = [], []
        step, done = 0, False

        # reset the environment first
        obs, _ = self.env.reset()

        # run the test
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _, _ = self.env.step(action)
            print(reward, done)
            step_list.append(step)
            reward_list.append(reward)

            if done:
                obs, _ = self.env.reset()

            step += 1
        print("Test Success.")

        # plot the reward
        plt.plot(step_list, reward_list)
        # save the plot
        plt.savefig(f"{self.log_dir}/test_reward_plot_{result_file}.png")
        # show the plot
        plt.show()

    def extract_args(self, args) -> Tuple[str, int, str, str]:
        # extract the arguments
        duty = args.duty
        env_type = args.env
        timesteps = args.timesteps
        model_dir = args.model_dir
        log_dir = args.log_dir

        return duty, env_type, timesteps, model_dir, log_dir

    def create_dir(
        self, model_name: str = "models", log_name: str = "logs"
    ) -> Tuple[str, str]:
        # create the folder from path
        os.makedirs(model_name, exist_ok=True)
        os.makedirs(log_name, exist_ok=True)

        return model_name, log_name

    def make_env(self, env_id: str, rank: int, seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """

        def _init() -> gym.Env:
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # helper function to wait for user input
    def wait_for_y(self) -> None:
        while True:
            user_input = input("Press 'Y' to continue: ")
            if user_input.upper() == "Y":
                print("Continuing process...\n\n")
                break
            else:
                print("Invalid input. Please press 'Y'.")
