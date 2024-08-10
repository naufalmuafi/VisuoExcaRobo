# VisuoExcaRobo

This repository contains VisuoExcaRobo i.e. Vision-based Control for Excavator with Deep-Reinforcement Learning.

## Project Description

VisuoExcaRobo is a reinforcement learning project using Webots, Stable Baselines3, and custom environment of Gymnasium. The goal is to develop a model that Excavator can visually recognize and move towards a target objects in simulated environment.

## Setup

### Prerequisites

- Python 3.9+
- Webots
- Virtual Environment (recommended)

### Setting Up Virtual Environment

1. **Clone the repository**:

   ```bash
   git clone https://github.com/naufalmuafi/VisuoExcaRobo.git
   cd VisuoExcaRobo
   ```

2. Create a virtual environment

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - **Windows**:

     ```bash
     .env\Scripts\activate
     ```

   - **Linux/MacOS**:

     ```bash
     source .env/bin/activate
     ```

4. Install the requirements:

    The project uses pip-tools for managing dependencies.
    - Install pip-tools:

      ```bash
      pip install pip-tools
      ```

    - Compile and install dependencies:

      ```bash
      pip-compile requirements.in
      pip install -r requirements.txt
      ```

### Setting Up PyCharm in Windows for a Webots Project

1. **Navigate to the Project Directory**:

   - Open Command Prompt and navigate to your Webots project directory.

     ```sh
     cd path\to\your\webots\project
     ```

2. **Create a Virtual Environment**:

   - Run the following command to create a virtual environment:

     ```sh
     python -m venv venv
     ```

3. **Launch PyCharm**:

   - Open PyCharm.

4. **Configure Project Structure**:

   - Go to `File > Settings > Project: <Your Project> > Project Structure`.
   - Click `Add Content Root` and select `WEBOTS_HOME/lib/controller/python`.

5. **Edit Configuration**:

   - Click `Edit Configurations` on the top right (next to the run/debug configurations dropdown).
   - Click the `+` icon and select `Python`.
   - In the `Name` field, give your configuration a name.
   - In the `Script` field, select your Python controller script.
   - In the `Environment variables` section, click `...` and then `+` to add a new environment variable.
     - Set the `Name` to `Path`.
     - Set the `Value` to `F:\Program Files\Webots\lib\controller\;F:\Program Files\Webots\msys64\mingw64\bin\;F:\Program Files\Webots\msys64\mingw64\bin\cpp`.
   - Make sure the interpreter is installed correctly.

6. **Develop Your Program**:

   - Start editing and developing your program within PyCharm.

7. **Install `pip-tools`**:

   - Open the terminal in PyCharm and run:

     ```sh
     pip install pip-tools
     ```

8. **Create `requirements.in` File**:

   - Create a file named `requirements.in` in your project directory.
   - List the required libraries in this file and save it.

9. **Compile and Sync Requirements**:

   - Run the following commands in the terminal:

     ```sh
     pip-compile
     pip-sync
     ```

10. **Run from IDE**:
    - In Webots, select the `<extern>` option to run the controller script from the IDE.

## Authors

[Naufal Mu'afi](nmuafi1@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
