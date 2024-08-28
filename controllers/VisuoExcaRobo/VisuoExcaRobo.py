# import Color_VisuoExcaRobo
# import Object_VisuoExcaRobo

import os
import datetime
import argparse
import gymnasium as gym
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


TIMESTEPS = 1000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4

# Create the parser
parser = argparse.ArgumentParser(
    description="Train and test the model with 2 options env: Color or Object"
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


class VisuoExcaRobo:
    def __init__(self) -> None:
        self.env_type, self.timesteps, self.model_name, self.log_name = (
            self.extract_args()
        )

        # Define the environment
        if self.env_type == "Color":
            self.env = gym.make("Color_VisuoExcaRobo-v1")
        elif self.env_type == "Object":
            self.env = gym.make("Object_VisuoExcaRobo-v1")

        # Create the directories
        self.model_dir, self.log_dir = self.create_dir(self.model_name, self.log_name)

    def extract_args(self) -> Tuple[str, int, str, str]:
        # parse the arguments
        args = parser.parse_args()

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

    def train_PPO(self) -> None:
        # use Proximal Policy Optimization (PPO) algorithm
        print("Training the model with PPO...")
        model = PPO(
            "CnnPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
        )

        # Get the current date in YYYYMMDD format
        today_date = datetime.datetime.now().strftime("%Y%m%d")

        # Construct the model filename
        model_filename = f"{self.model_dir}/{today_date}_ppo_{self.timesteps}_bs_{BATCH_SIZE}_lr_{LEARNING_RATE:.0e}"

        # Train and save the model
        model.learn(total_timesteps=self.timesteps)
        model.save(model_filename)
        print(f"Model saved as {model_filename}")

        return model_filename

    def test_PPO(self, steps: int = 500, model_file: str = "models") -> None:
        # load the model
        try:
            model = PPO.load(model_file, env=self.env)
        except FileNotFoundError:
            print(
                "Model not found. Please train the model first/type your model name correctly."
            )
            return

        print("Load Model Successful")

        # run a test
        obs, _ = self.env.reset()
        for i in range(steps):
            action, _states = model.predict(obs)
            obs, reward, done, _, _ = self.env.step(action)
            print(reward, done)

            if done:
                obs, _ = self.env.reset()
                print("Test Terminated.")

            if i == steps - 1:
                print("Test Successful")

    # helper function to wait for user input
    def wait_for_y(self) -> None:
        while True:
            user_input = input("Press 'Y' to continue: ")
            if user_input.upper() == "Y":
                print("Continuing process...\n\n")
                break
            else:
                print("Invalid input. Please press 'Y'.")


# main program
if __name__ == "__main__":
    # Instantiate the VisuoExcaRobo class
    ver = VisuoExcaRobo()

    # Check the environment
    ready = ver.check_environment()

    if ready:
        print(f"Environment is ready: {ver.env}")

        # Train the model
        model_file = ver.train_PPO()

        print("Training is finished, press `Y` for replay...")
        ver.wait_for_y()

        # Test the environment
        print("Testing the Environment with Predicted Value")
        ver.test_PPO(model_file=model_file)
