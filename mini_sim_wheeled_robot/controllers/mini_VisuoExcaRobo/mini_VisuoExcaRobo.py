import os
import env_VisuoExcaRobo
import gymnasium as gym
from typing import Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


TIMESTEPS = 1000

# where to store trained model and logs
model_dir_name = "models_mini"
log_dir_name = "logs_mini"

# define the environment
env = gym.make("mini_VisuoExcaRobo-v1")


def create_dir(model_name: str = "models", log_name: str = "logs") -> Tuple[str, str]:
    # create the folder from path
    os.makedirs(model_name, exist_ok=True)
    os.makedirs(log_name, exist_ok=True)

    return model_name, log_name


def check_environment(env: gym.Env) -> None:
    # check the environment
    try:
        print(f"Check the environment: {env}...")
        check_env(env)
    except Exception as e:
        print(f"Environment check failed: {e}")
        print("Please check the environment and try again.")
        exit(1)


def train_PPO(env: gym.Env, model_dir: str, log_dir: str, timesteps) -> None:
    # use Proximal Policy Optimization (PPO) algorithm
    # use MLP policy for observation space 1D-vector
    print("Training the model with PPO...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
    # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)

    # train and save the model
    model.learn(total_timesteps=timesteps)
    model.save(f"{model_dir}/ppo_{timesteps}")


def test_PPO(env: gym.Env, model_dir: str, timesteps: int = TIMESTEPS) -> None:
    # load the model
    try:
        model = PPO.load(f"{model_dir}/ppo_{timesteps}", env=env)
    except FileNotFoundError:
        print("Model not found. Please train the model first.")
        return

    print("Load Model Successful") if model else None

    # run a test
    obs, _ = env.reset()
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        print(obs, reward, done)

        if done:
            obs, _ = env.reset()
            print("Test Terminated.")

        print("Test Successful") if i == timesteps else None


# helper function to wait for user input
def wait_for_y() -> None:
    while True:
        user_input = input("Press 'Y' to continue: ")
        if user_input.upper() == "Y":
            print("Continuing process...\n\n")
            break
        else:
            print("Invalid input. Please press 'Y'.")


# main program
if __name__ == "__main__":
    # create directories
    model_dir, log_dir = create_dir(model_dir_name, log_dir_name)

    # check the environment
    check_environment(env)
    print(f"Environment is ready: {env}")

    # train and test the model with A2C algorithm
    train_PPO(env, model_dir, log_dir, TIMESTEPS)

    print("Training is finished, press `Y` for replay...")
    wait_for_y()
    print("Test the Environment with Predicted Value")

    test_PPO(env, model_dir, TIMESTEPS)
