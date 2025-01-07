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

## Supported Models

### Machine Learning Models
- **BERT**: Sequence classification for ad interaction prediction.
- **XLNet**: Contextual sequence modeling for sequential user behavior.
- **LSTM, GRU**: Time-series prediction for sequential ad clicks and views.
- **CNN**: Pattern detection in user interaction sequences.

### Reinforcement Learning
- **LAVA**: Optimized for offline RL tasks with latent action spaces.
- **Decision Transformer**: Sequence modeling for decision-making processes.
- **MuZero**: Dynamic modeling for sequential decision-making in unknown environments.

### Generative and Multimodal Models
- **Diffusion Models**: NLP-based text generation for ad interactions.
- **CLIP**: Multimodal learning for text-image alignment in advertisements.
- **GTN (Generative Teaching Networks)**: Synthetic data generation for user interaction simulations.


## Synthetic Data Creation

The **AdFlux Engine** includes utilities to generate synthetic datasets for simulating ad interaction environments. This is particularly useful when real-world data is unavailable or limited.

### `SyntheticAdDataset` Class

The synthetic data generator produces sequences representing user behavior or ad interactions. These sequences can be used for training and testing models for tasks like:

- Predicting user clicks or views.
- Simulating sequential ad interactions in RL environments.


## Getting Started

### Prerequisites
- **Python**: Version 3.8 or higher.
- **GPU**: A CUDA-enabled GPU is recommended for training large models.
- **Dependencies**: Install the required libraries:
  ```bash
  pip install -r requirements.txt
  ```

  ## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/adflux-engine.git
   cd adflux-engine
   ```
2. Ensure the directory structure matches the above layout.
3. Prepare data:
- **Place real-world or synthetic datasets** in the `data/` directory.
- **Use the `DataProcessor` class** to preprocess and split the data.


### Switch Models Configuration Update

To switch to a different model, update the `config.py` file with the new model configuration.
```python
MODEL_CONFIG = {
    'current_model': 'decision_transformer'  # Change to 'lava', 'gato', etc.
}
```

- Example Models:
'lava'
'gato'
'decision_transformer'


  ## Usage

### Run the Main Script

Train and evaluate a model on ad interaction data:

```bash
python main.py
```


### Simulate Ad Interactions

Use RL models to predict user behavior in an ad environment:
```bash
python rl/ads_seq_prediction_rl.py
```

### Generate Synthetic Data

Create synthetic ad interaction datasets using the SyntheticAdDataset class.
```bash
This markdown version properly formats the code blocks and headings for your `.md` file.
```


## Test Models

Run unit tests for models:

```bash
python tests/test_models.py
```

Running a Simulation with the Decision Transformer
```bash
python main.py --model decision_transformer --data data/user_data.csv
```
Training with LAVA
```bash
python main.py --model lava --train --data data/user_data.csv
```

## Future Enhancements

- **Incorporate AutoML**: Automatically select the best model and hyperparameters for your dataset.
- **Expand Reward Functions**: Introduce more sophisticated reward mechanisms for RLHF.
- **Add Visualization Tools**: Provide dashboards to analyze predictions and user behaviors.


