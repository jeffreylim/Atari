#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym

from exploratory_scheduler import ExploratoryScheduler
from memories import ReplayExperiences
from dqn import DuelDQN_Deepmind, DuelDQN_Deepmind

from baselines.common.atari_wrappers import make_atari, wrap_deepmind


def train(env, target_success_threshold, model_name):

    learning_rate = 0.00001  # 0.00025
    discount_factor = 0.99

    update_main_model_freq = 4
    update_target_model_freq = 10000

    main_dqn = DuelDQN_Deepmind(learning_rate, env.action_space.n)
    target_dqn = DuelDQN_Deepmind(learning_rate, env.action_space.n)

    decay_epsilon_min = 0.1
    decay_epsilon_max = 1.0
    decay_max_steps = 1000000
    max_exploration_actions = 50000
    exploratory_scheduler = ExploratoryScheduler(max_exploration_actions, decay_epsilon_max, decay_epsilon_min,
                                                 decay_max_steps)

    max_memory_capacity = 100000
    memory_batch_size = 32
    replay_experiences = ReplayExperiences(max_memory_capacity, memory_batch_size)

    max_episodic_reward_history = 100
    episodic_reward_history = np.empty(max_episodic_reward_history, dtype=np.float32)

    running_reward = 0
    episode_count = 0
    frame_count = 0
    max_frames_per_episode = 10000
    max_total_frames = 5*1e7 

    while frame_count < max_total_frames:

        state = np.array(env.reset())
        episode_reward = 0

        for _ in range(max_frames_per_episode):

            frame_count += 1

            if exploratory_scheduler(frame_count):
                action = np.random.choice(env.action_space.n)
            else:
                state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
                q_values = main_dqn.model(state_tensor, training=False)
                action = tf.argmax(q_values[0]).numpy()

            next_state, reward, done, _ = env.step(action)

            next_state = np.array(next_state)

            replay_experiences.add(action, state, next_state, reward, done)

            episode_reward += reward

            state = next_state

            if frame_count % update_main_model_freq == 0 and frame_count > memory_batch_size:
                actions, states, next_states, rewards, dones = replay_experiences.sample()

                q_values = target_dqn.model.predict(next_states)
                discounted_cum_q_values = rewards + discount_factor * tf.reduce_max(q_values, axis=1)
                discounted_cum_q_values = discounted_cum_q_values * (1 - dones) - dones
                masks = tf.one_hot(actions, env.action_space.n)

                with tf.GradientTape() as tape:
                    q_values = main_dqn.model(states)
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = keras.losses.Huber()(discounted_cum_q_values, q_action)

                grads = tape.gradient(loss, main_dqn.model.trainable_variables)
                main_dqn.optimizer.apply_gradients(zip(grads, main_dqn.model.trainable_variables))

            if frame_count % update_target_model_freq == 0:
                target_dqn.model.set_weights(main_dqn.model.get_weights())
                print('Running mean reward: {:.3f}, Episode: {}, Frame: {}'.format(running_reward, episode_count,
                                                                                   frame_count))

            if done:
                break

        episode_count += 1
        episodic_reward_history[episode_count % max_episodic_reward_history] = episode_reward
        running_reward = np.mean(episodic_reward_history[:min(episode_count, max_episodic_reward_history)])
        print('Frame: {}, Episode: {}, Episode reward: {:.3f}'.format(frame_count, episode_count, episode_reward))

        if running_reward > target_success_threshold:
            print('Target reached at episode: {}, frame: {}, running_reward: {:.3f}'.format(episode_count, frame_count,
                                                                                            running_reward))
            break

    model_name += ('_' + str(running_reward) + '.h5')
    main_dqn.model.save(model_name)
    return running_reward, keras.models.load_model(model_name)


def evaluate(env, eval_model, output_dir, num_of_evals=10):

    env = gym.wrappers.Monitor(env, output_dir, force=True)

    for _ in range(num_of_evals):

        obs = env.reset()

        total_reward = 0
        step_count = 0

        while True:

            step_count += 1

            env.render()

            obs_t = tf.convert_to_tensor(obs)
            obs_t = tf.expand_dims(obs_t, 0)

            action_values = trained_model.predict(obs_t)
            action = tf.argmax(action_values[0]).numpy()

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                print('Total reward: {:.3f} in {} steps.'.format(total_reward, step_count))
                break


if __name__ == '__main__':

    ENV_NAME = 'BreakoutNoFrameskip-v4'

    # make_atari enables NoopResetEnv and MaxAndSkipEnv
    env = make_atari(ENV_NAME)

    env = wrap_deepmind(env,
                        episode_life=True,
                        clip_rewards=True,
                        frame_stack=True,
                        scale=True)

    seed = 22
    env.seed(seed)

    target_success_threshold = 50
    model_name = ENV_NAME

    running_reward, trained_model = train(env, target_success_threshold, model_name)

    output_dir = './' + model_name + '_' + str(running_reward)
    evaluate(env, trained_model, output_dir)

    env.close()
