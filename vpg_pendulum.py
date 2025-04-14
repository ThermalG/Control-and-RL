"""EEME E6602 Project
Author @AlexWei
Last modified: 04/10/2025

ACADEMIC INTEGRITY STATEMENT: partial implementations especially train() & discount_rewards() are transplanted / adapted based on an earlier project of mine in EECS E6892
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm

env = gym.make('Pendulum-v1')
obs_size, act_size = env.observation_space.shape[0], env.action_space.shape[0]
low, high = env.action_space.low, env.action_space.high

def actor_creator(state_dim, action_dim, bound, log_std_init=-0.5):
    """Creates an actor model suitable for continuous action spaces in reinforcement learning;
    Outputs actions based on a mean policy and has a learnable logarithm of the standard deviation
    """
    state_input = layers.Input(shape=(state_dim,))

    hidden1 = layers.Dense(64, activation='relu')(state_input)
    hidden2 = layers.Dense(64, activation='relu')(hidden1)

    mu_output = layers.Dense(action_dim, activation='tanh')(hidden2)
    mu_scaled = layers.Lambda(lambda x: x * bound)(mu_output)

    # output log standard deviation directly, using a Lambda layer
    log_std = tf.Variable(initial_value=np.full((action_dim,), log_std_init), dtype=tf.float32, trainable=True)
    log_std_output = tf.tile(tf.expand_dims(log_std, axis=0), (tf.shape(state_input)[0], 1)) # matching batch size

    return models.Model(inputs=state_input, outputs=[mu_scaled, log_std_output])

def critic_creator(state_dim):
    state_input = layers.Input(shape=(state_dim,))

    hidden1 = layers.Dense(64, activation='relu')(state_input)
    hidden2 = layers.Dense(64, activation='relu')(hidden1)
    value_output = layers.Dense(1, activation=None)(hidden2)    # output layer for state value estimation

    return models.Model(inputs=state_input, outputs=value_output)

def sample_traj(mdl, batch=2000, seed=None):
    """Samples trajectories from the environment using the provided actor model

    Parameters:
    - mdl: Actor model in use to sample actions
    - batch (int): The number of states visited

    Returns:
    - states, actions, rewards, not_dones, and average episodic reward
    """
    s, a, r, not_dones = [], [], [], []
    curr_reward_list = []
    collected = 0
    env_seed = seed

    # continue sampling until reaching the specified batch size
    while collected < batch:
        state = env.reset(seed=env_seed)[0] # reset env at the start or after each ep ends
        if env_seed is not None:
            env_seed = None

        curr_reward = 0
        terminated, truncated = False, False

        # sample actions from the actor and step through env until end
        while not (terminated or truncated):
            # prepare current state for the actor model
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            mean, log_std = mdl(state_tensor, training=False)
            std = tf.exp(log_std)

            # sample an action from Gaussian dist
            action = tf.clip_by_value(mean + tf.random.normal(tf.shape(mean)) * std, low, high)
            action = action.numpy().flatten()

            # execute action in env to get the next state & reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            s.append(state)
            a.append(action)
            r.append(reward)
            not_dones.append(0.0 if done else 1.0)

            state = next_state
            curr_reward += reward
            collected += 1

            if done:
                break

        curr_reward_list.append(curr_reward)

    return np.array(s, dtype=np.float32), np.array(a, dtype=np.float32), np.array(r, dtype=np.float32), np.array(not_dones, dtype=np.float32), np.mean(curr_reward_list)


def discount_rewards(reward_buffer, dones, gamma):
    g_t = np.zeros_like(reward_buffer, dtype=float)
    running_add = 0
    num_traj = 0
    for t in reversed(range(len(reward_buffer))):
        running_add = reward_buffer[t] + gamma * running_add * dones[t]
        g_t[t] = running_add

        # reset accumulator and count number of traj at the end of each ep
        if dones[t] == 0:
            num_traj += 1
        if len(dones) > 0 and dones[-1] != 0:
         num_traj += 1 # account for the last trajectory

    # for edge case of an empty buffer
    if len(reward_buffer) == 0:
        num_traj = 0

    return g_t.astype(np.float32), max(1, num_traj)

def train(model_actor, model_critic, opt_actor, opt_critic, s, a, r, dones, gamma):
    s = tf.convert_to_tensor(s, dtype=tf.float32)
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    g_t, _ = discount_rewards(r.numpy(), dones.numpy(), gamma)
    g_t = tf.convert_to_tensor(g_t, dtype=tf.float32)

    with tf.GradientTape() as tape: # update critic model
        critics = model_critic(s, training=True)
        critics = tf.squeeze(critics, axis=1)
        loss_critic =tf.keras.losses.mean_squared_error(g_t, critics)

    critic_grads = tape.gradient(loss_critic, critic.trainable_variables)
    opt_critic.apply_gradients(zip(critic_grads, critic.trainable_variables))

    with tf.GradientTape() as tape: # update actor model
        # log probabilities
        means, log_stds = model_actor(s, training=True)
        neg_log_prob = -0.5 * tf.reduce_sum(tf.square((a - means) / (tf.exp(log_stds) + 1e-8)), axis=1)
        neg_log_prob -= 0.5 * tf.cast(tf.shape(a)[1], tf.float32) * tf.math.log(2.0 * np.pi)
        neg_log_prob -= tf.reduce_sum(log_stds, axis=1)

        # compute and normalize the advantages tensor
        # A2C reference: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
        advantages = g_t - tf.stop_gradient(tf.squeeze(model_critic(s, training=False), axis=1))
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        # compute loss based on policy gradient estimate
        loss_actor = -tf.reduce_mean(neg_log_prob * advantages)

    actor_grads = tape.gradient(loss_actor, model_actor.trainable_variables)
    opt_actor.apply_gradients(zip(actor_grads, model_actor.trainable_variables))

    return loss_critic.numpy(), loss_actor.numpy()

# tunable hyperparams
GAMMA = 0.99            # discount factor
last_n_reward = 100     # number of episodes for calculating running reward
TRAIN_EPISODES = 2000
actor_lr = 3e-4
critic_lr = 1e-3
batch_size = 5000

# initialization
actor = actor_creator(obs_size, act_size, high)
critic = critic_creator(obs_size)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
episode_reward_history = []
running_rewards, episode_rewards = [], []
actor_losses, critic_losses = [], []
consistency = 0

# main loop
pbar = tqdm(range(TRAIN_EPISODES))
for ep in pbar:
    # sample trajectories using current policy
    states, actions, rewards, n_dones, episode_reward = sample_traj(actor, batch=batch_size)

    # update actor & critic models using the sampled trajectories
    critic_loss, actor_loss = train(actor, critic, actor_optimizer, critic_optimizer, states, actions, rewards, n_dones, GAMMA)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)

    episode_reward_history.append(episode_reward)
    episode_rewards.append(episode_reward)

    if len(episode_reward_history) > last_n_reward: # keep only the last n rewards
        del episode_reward_history[0]

    running_reward = np.mean(episode_reward_history)
    running_rewards.append(running_reward)

    pbar.set_postfix(EpisodeReward=f'{episode_reward:.2f}', RunningReward=f'{running_reward:.2f}')

    if episode_reward >= -200:  # early stopping if diverged well
        consistency += 1
        if consistency >= 10:
            print("Early stopping at episode {ep} since target achieved.")
            break
    else:
        consistency = 0

pbar.close()

actor.save_weights("vpg_actor_weights.h5")
critic.save_weights("vpg_critic_weights.h5")

# visualize training progress: Appendix A3
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(running_rewards, label="running reward")
plt.plot(episode_rewards, label="episode reward", alpha=0.4)
plt.xlabel("episode")
plt.ylabel("running reward (last 100 ep)")
plt.title("TRAINING RESULTS - Evolution")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(actor_losses, label="actor loss")
plt.plot(critic_losses, label="critic loss")
plt.xlabel("episode")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# pendulum balancing visualization: Appendix A4
actor.load_weights("vpg_actor_weights.h5")

env = gym.make("Pendulum-v1", render_mode="human")  # separate env for testing
obs = env.reset()[0]

done = False
while not done:
    mu, _ = actor(tf.convert_to_tensor(np.expand_dims(obs, axis=0), dtype=tf.float32), training=False)
    action = tf.clip_by_value(mu, env.action_space.low, env.action_space.high).numpy().flatten()

    obs, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()