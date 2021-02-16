import numpy as np

from decay_factor import LinearDecayFactor


class ExploratoryScheduler:

    def __init__(self, max_exploration_actions, decay_epsilon_max, decay_epsilon_min, decay_max_steps):
        self.max_exploration_actions = max_exploration_actions
        self.decay_factor = LinearDecayFactor(decay_epsilon_max, decay_epsilon_min, decay_max_steps)

    def __call__(self, frame_count):
        if frame_count < self.max_exploration_actions or self.decay_factor(frame_count) > np.random.uniform():
            return True
        return False
