import torch
import torch.nn as nn
from learn2learn.algorithms import MAML

class MetaLearningModel:
    """
    Meta-Learning framework using MAML for rapid adaptation to new tasks.
    """
    def __init__(self, model, lr=0.01):
        """
        Initializes the Meta-Learner.
        """
        self.inner_model = model
        self.meta_learner = MAML(self.inner_model, lr=lr)

    def adapt(self, task_data, loss_function):
        """
        Adapts the model to a specific task using task data.
        """
        adapted_learner = self.meta_learner.clone()
        task_inputs, task_labels = task_data

        for _ in range(5):  # Number of inner loop steps
            predictions = adapted_learner(task_inputs)
            loss = loss_function(predictions, task_labels)
            adapted_learner.adapt(loss)

        return adapted_learner

    def evaluate(self, task_data, loss_function):
        """
        Evaluates the adapted model on unseen data.
        """
        task_inputs, task_labels = task_data
        predictions = self.inner_model(task_inputs)
        loss = loss_function(predictions, task_labels)
        accuracy = (predictions.argmax(dim=-1) == task_labels).float().mean()
        return {"loss": loss.item(), "accuracy": accuracy.item()}

