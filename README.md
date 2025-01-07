# AdFlux Engine

![Screenshot_2025-01-06_at_8 42 56_AM-removebg-preview](https://github.com/user-attachments/assets/c7ecef31-f692-4a0b-9bea-d79ca6c696ea)


## Overview
AdFlux Engine is a Advertisement Simulator designed to predict and simulate user behavior (clicks and views) based on historical interaction data. Leveraging cutting-edge architectures like LAVA, Decision Transformers, Gato, and MuZero, AdFlux Engine helps optimize ad placements, enhance user engagement, and maximize conversion rates.

Built with modularity and scalability in mind, AdFlux Engine combines the latest advancements in NLP, reinforcement learning, and sequence modeling, including RLHF (Reinforcement Learning from Human Feedback).

## Features

- **Multi-Model Support**: Use the latest SOTA models, such as LAVA, Decision Transformer, and Gato, or easily add new architectures.
- **Reinforcement Learning with RLHF**: Train models to align with business goals using offline reinforcement learning.
- **Sequence Modeling**: Accurately predict user behavior using advanced transformer-based models like TimeSformer.
- **Customizable Framework**: Modular structure for easy customization and expansion.
- **Simulations & Analytics**: Run simulations to test ad strategies and analyze results.

## Getting Started

Prerequisites
Python 3.8 or higher
Recommended: CUDA-enabled GPU for training large models
Install Python dependencies:
pip install -r requirements.txt
Installation
Clone this repository:
git clone https://github.com/your-username/ad-simulator.git
cd ad-simulator
Set up a Python virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Usage
Prepare Data: Add your user interaction data in the data/ directory.
Supported formats: .csv, .json
Use data/data_processor.py to preprocess the data.
Run Simulations:
Use main.py to start a simulation:
python main.py
Switch Models:
Update the configuration in config.py to use a different model:
MODEL_CONFIG = {
    'current_model': 'decision_transformer'  # Change to 'lava', 'gato', etc.
}

## Models Supported

1. LAVA (Latent Action Spaces for Offline RL)
Optimizes for offline reinforcement learning tasks using latent action representations.
Use Case: High-performance advertisement strategies from historical data.
2. Decision Transformer
Combines transformer-based sequence modeling with reinforcement learning.
Use Case: Predicts optimal user action sequences.
3. Gato
A generalist model capable of handling text, images, and control tasks.
Use Case: Cross-domain user interaction modeling.
4. MuZero
Learns a model of the environment for dynamic decision-making.
Use Case: Simulating complex, long-term user behavior.
5. TimeSformer
A transformer model specialized for time-series forecasting.
Use Case: Predicting user behavior trends over time.

## Examples

Running a simulation with the Decision Transformer:

python main.py --model decision_transformer --data data/user_data.csv
Training with LAVA:

python main.py --model lava --train --data data/user_data.csv

## Future Work

Incorporate AutoML: Automatically select the best model and hyperparameters for your dataset.
Expand Reward Functions: Introduce more sophisticated reward mechanisms for RLHF.
Add Visualization Tools: Provide dashboards to analyze predictions and user behaviors.
