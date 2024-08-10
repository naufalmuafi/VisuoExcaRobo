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

## Authors

[Naufal Mu'afi](nmuafi1@gmail.com)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
