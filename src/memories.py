import numpy as np


class ReplayExperiences:

    def __init__(self, capacity=1000000, batch_size=32):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer_idx = 0
        self.num_experiences = 0

        self.actions = np.empty(self.capacity, dtype=np.int32)
        self.states = np.empty(self.capacity, dtype=np.ndarray)
        self.next_states = np.empty(self.capacity, dtype=np.ndarray)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.episode_status = np.empty(self.capacity, dtype=np.float32)

    def add(self, action, state, next_state, reward, done):
        self.actions[self.buffer_idx] = action
        self.states[self.buffer_idx] = state
        self.next_states[self.buffer_idx] = next_state
        self.rewards[self.buffer_idx] = reward
        self.episode_status[self.buffer_idx] = done
        self.buffer_idx = (self.buffer_idx + 1) % self.capacity
        self.num_experiences = min(self.capacity, self.num_experiences + 1)

    def sample(self):
        indices = np.random.choice(range(self.num_experiences), size=self.batch_size)

        actions = np.array([self.actions[i] for i in indices])
        states = np.array([self.states[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        dones = np.array([self.episode_status[i] for i in indices])

        return actions, states, next_states, rewards, dones

