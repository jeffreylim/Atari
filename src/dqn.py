import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DuelDQN:

    def __init__(self, learning_rate, action_space_dim):

        inputs = layers.Input(shape=(84, 84, 4,))

        layer1 = layers.Conv2D(filters=32, kernel_size=[8, 8], strides=4, activation='relu')(inputs)
        layer2 = layers.Conv2D(filters=64, kernel_size=[4, 4], strides=2, activation='relu')(layer1)
        layer3 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, activation='relu')(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(filters=512, activation='relu')(layer4)

        state_values = keras.layers.Dense(1)(layer5)
        raw_advantages = layers.Dense(action_space_dim, activation='linear')(layer5)

        advantages = raw_advantages - keras.backend.max(raw_advantages, axis=1, keepdims=True)
        q_values = state_values + advantages

        self.model = keras.Model(inputs=inputs, outputs=q_values)

        self.optimizer = keras.optimizers.Adam(learning_rate, clipnorm=1.0)


class DuelDQN_Deepmind:

    def __init__(self, learning_rate, action_space_dim):

        inputs = layers.Input(shape=(84, 84, 4,))

        layer1 = layers.Conv2D(filters=32, kernel_size=[8, 8], strides=4, use_bias=False,
                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                               activation='relu')(inputs)

        layer2 = layers.Conv2D(filters=64, kernel_size=[4, 4], strides=2, use_bias=False,
                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                               activation='relu')(layer1)

        layer3 = layers.Conv2D(filters=64, kernel_size=[3, 3], strides=1, use_bias=False,
                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                               activation='relu')(layer2)

        layer4 = layers.Conv2D(filters=1024, kernel_size=[7, 7], strides=1, use_bias=False,
                               kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                               activation='relu')(layer3)

        valuestream, advantagestream = tf.split(layer4, 2, 3)
        valuestream = layers.Flatten()(valuestream)
        advantagestream = layers.Flatten()(advantagestream)

        advantage = layers.Dense(units=action_space_dim,
                                 kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                                 name='advantage')(advantagestream)
        value = layers.Dense(units=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2),
                             name='value')(valuestream)
        q_values = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keepdims=True))

        self.model = keras.Model(inputs=inputs, outputs=q_values)

        self.optimizer = keras.optimizers.Adam(learning_rate, clipnorm=1.0)

