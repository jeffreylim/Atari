import matplotlib.pyplot as plt


class LinearDecayFactor:

    def __init__(self, epsilon_max=1.0, epsilon_min=0.1, max_steps=1000000):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.max_steps = max_steps

        self.decay_factor = (self.epsilon_min - self.epsilon_max) / (self.max_steps)

    def __call__(self, current_step_count):
        self.epsilon_decay = self.epsilon_max + self.decay_factor * current_step_count
        return max(self.epsilon_min, self.epsilon_decay)

    def get_decay_factor(self):
        return max(self.epsilon_min, self.epsilon_decay)


if __name__ == '__main__':

    epsilon_start = 1.0
    epsilon_end = 0.1
    max_steps = 15
    episode_length = 20

    decay = LinearDecayFactor(epsilon_start, epsilon_end, max_steps)
    epsilon = [decay(step) for step in range(episode_length)]

    plt.plot(epsilon)
    plt.show()