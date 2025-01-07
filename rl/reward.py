class RewardCalculator:
    """
    Reward function for RL tasks in ad simulations.
    """
    def __init__(self):
        pass

    def calculate_reward(self, action, success=True):
        """
        Computes reward based on action and success metric.
        """
        if success:
            return 10 if action == 1 else 5  # Reward for specific actions
        else:
            return -5  # Penalty for unsuccessful actions

