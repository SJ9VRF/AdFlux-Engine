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


## Supported Models

### Machine Learning Models
- **BERT**: Captures bidirectional context in ad interactions, enabling accurate predictions of user engagement or intent. Encodes human feedback as contextual input, improving reward modeling for better user-aligned policies.
- **XLNet**: Leverages permutation-based training to better understand the order of ad interactions. Provides enhanced sequence modeling for feedback-driven reward optimizations.
- **LSTM, GRU**: Handles long-term dependencies in sequential ad clicks and views, making it suitable for time-series analysis. Captures trends in human feedback over time for policy refinement.
- **CNN**: Identifies patterns in user interaction sequences, such as spikes in engagement or drop-offs. Quickly learns localized features in feedback data, speeding up alignment with human preferences.

### Reinforcement Learning Models
- **LAVA**: Efficiently models offline ad interactions to recommend optimal ad placements without requiring live interaction data. Learns compact latent representations, allowing better integration of complex human feedback.
- **Decision Transformer**: Models sequential decision-making as a sequence generation task, optimizing ad sequences for maximum user engagement. Integrates feedback-driven goals into sequence generation, ensuring alignment with user expectations.
- **MuZero**: Models dynamic ad environments, accounting for unseen user behavior patterns and optimizing sequences adaptively. Combines planning and learned policies to improve long-term user satisfaction through human-aligned actions.

### Generative and Multimodal Models
- **Diffusion Models**: Generates realistic ad interaction sequences, useful for training models in data-scarce scenarios. Generates synthetic feedback data to augment RL training for better policy alignment.
- **CLIP (Multimodal Learning)**: Aligns text and image features to create visually cohesive and contextually relevant ad content. Incorporates multimodal feedback, improving alignment with user preferences across different media formats.
- **GTN (Generative Teaching Networks)**: Produces synthetic training data tailored to model requirements, reducing dependency on real-world data. Generates high-quality, diverse feedback data for efficient policy refinement.

### Efficient Transformers
- **Linformer**: Scales transformer models to handle long ad sequences with reduced computational overhead. Enables real-time feedback modeling by handling large-scale interaction data efficiently.
- **Performer**: Provides linear attention for fast and accurate modeling of ad interactions over time. Processes human feedback at scale with low latency, improving response times in adaptive systems.
- **Reformer**: Reduces memory usage while handling long ad interaction sequences, making it suitable for resource-constrained environments. Handles detailed feedback logs efficiently, enabling scalable policy updates.

### Specialized Models
- **Capsule Networks**: Captures hierarchical relationships in ad interactions, such as grouping ads by category or campaign. Models structured human feedback, improving reward function accuracy.
- **Meta-Learning Models**: Adapts quickly to new ad campaigns or user behaviors with minimal retraining. Learns how to generalize from limited feedback data, accelerating alignment processes.



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


### Switch Models

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


## More Advanced Models
- **[AdFlux PersonaTaste Engine](https://github.com/SJ9VRF/AdFlux-PersonaTaste-Engine)**: Enhancing AdFlux by incorporating persona-based tastes, utilizing user behavior history such as shopping data or web clicks.
- **[AdFlux Agentic Engine](https://github.com/SJ9VRF/AdFlux-Agentic-Engine)**: An agentic engine designed to address the complexities of a sophisticated ad system.


