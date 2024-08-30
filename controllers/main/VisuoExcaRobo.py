import Color_VisuoExcaRobo
import Object_VisuoExcaRobo

import os
import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class VisuoExcaRobo:
    def __init__(self, args) -> None:
        self.env_type, self.timesteps, self.model_name, self.log_name = (
            self.extract_args(args)
        )

        # Define the environment
        if self.env_type == "Color":
            self.env = gym.make("Color_VisuoExcaRobo")
        elif self.env_type == "Object":
            self.env = gym.make("Object_VisuoExcaRobo")

        # Create the directories
        self.model_dir, self.log_dir = self.create_dir(self.model_name, self.log_name)

    def extract_args(self, args) -> Tuple[str, int, str, str]:
        # extract the arguments
        env_type = args.env
        timesteps = args.timesteps
        model_dir = args.model_dir
        log_dir = args.log_dir

        return env_type, timesteps, model_dir, log_dir

    def create_dir(
        self, model_name: str = "models", log_name: str = "logs"
    ) -> Tuple[str, str]:
        # create the folder from path
        os.makedirs(model_name, exist_ok=True)
        os.makedirs(log_name, exist_ok=True)

        return model_name, log_name

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

    def train_PPO(self, batch_size=64, learning_rate=0.0003) -> None:
        # Get the current date in YYYYMMDD format
        today_date = datetime.datetime.now().strftime("%Y%m%d")

        # Construct the model filename
        model_filename = f"{self.model_dir}/{today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"
        log_filename = f"{self.log_dir}/{today_date}_ppo_{self.timesteps}_bs_{batch_size}_lr_{learning_rate:.0e}"

        # use Proximal Policy Optimization (PPO) algorithm
        print("Training the model with PPO...")
        model = PPO(
            "CnnPolicy",
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

        return model_filename

    def test_PPO(self, steps: int = 500, model_file: str = None) -> None:
        # load the model
        try:
            model = PPO.load(model_file, env=self.env)
        except FileNotFoundError:
            print(
                "Model not found. Please train the model first/type your model name correctly."
            )
            return

        print("Load Model Successful")
        
        step_list, reward_list = [], []

        # run a test
        obs, _ = self.env.reset()
        for i in range(steps):
            action, _states = model.predict(obs)
            obs, reward, done, _, _ = self.env.step(action)
            print(reward, done)
            step_list.append(i)
            reward_list.append(reward)

            if done:
                obs, _ = self.env.reset()
                print("Test Terminated.")

            if i == steps - 1:
                print("Test Successful")
        
        # plot the reward
        plt.plot(step_list, reward_list)
        
        # save the plot
        plt.savefig(f"{self.log_dir}/reward_plot.png")
        
        # show the plot
        plt.show()

    # helper function to wait for user input
    def wait_for_y(self) -> None:
        while True:
            user_input = input("Press 'Y' to continue: ")
            if user_input.upper() == "Y":
                print("Continuing process...\n\n")
                break
            else:
                print("Invalid input. Please press 'Y'.")
