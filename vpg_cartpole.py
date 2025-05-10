"""EEME E6602 Project
Author @AlexWei
Last modified: 05/01/2025

ACADEMIC INTEGRITY STATEMENT: partial implementations especially train() & discount_rewards() are transplanted / adapted based on an earlier project of mine in EECS E6892
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU') # CPU generally 80% faster than GPU
env = gym.make('CartPole-v1')
obs_size = env.observation_space.shape[0]  # 4: segway pos, segway vel, body angle, body vel
act_size = env.action_space.n  # 2: [left, right]

def actor_creator(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    hidden1 = layers.Dense(64, activation='relu')(state_input)
    hidden2 = layers.Dense(64, activation='relu')(hidden1)
    logits = layers.Dense(action_dim)(hidden2)  # output logits for each action
    return models.Model(inputs=state_input, outputs=logits)

def critic_creator(state_dim):
    state_input = layers.Input(shape=(state_dim,))
    hidden1 = layers.Dense(64, activation='relu')(state_input)
    hidden2 = layers.Dense(64, activation='relu')(hidden1)
    value_output = layers.Dense(1, activation=None)(hidden2)
    return models.Model(inputs=state_input, outputs=value_output)

def sample_traj(mdl, batch=2000, seed=None):
    s, a, r, not_dones = [], [], [], []
    curr_reward_list = []
    collected = 0
    env_seed = seed

    while collected < batch:
        state = env.reset(seed=env_seed)[0]
        if env_seed is not None:
            env_seed = None
        curr_reward = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            logits = mdl(state_tensor, training=False)
            probs = tf.nn.softmax(logits).numpy()[0]
            action = np.random.choice(act_size, p=probs)  # sample action from probabilities
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

    return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32), np.array(not_dones, dtype=np.float32),
            np.mean(curr_reward_list))

def discount_rewards(reward_buffer, dones, gamma):
    g_t = np.zeros_like(reward_buffer, dtype=float)
    running_add = 0
    num_traj = 0
    for t in reversed(range(len(reward_buffer))):
        running_add = reward_buffer[t] + gamma * running_add * dones[t]
        g_t[t] = running_add
        if dones[t] == 0:
            num_traj += 1
    if len(dones) > 0 and dones[-1] != 0:
        num_traj += 1
    if len(reward_buffer) == 0:
        num_traj = 0
    return g_t.astype(np.float32), max(1, num_traj)

# adapted for discrete actions
def train(model_actor, model_critic, opt_actor, opt_critic, s, a, r, dones, gamma):
    s = tf.convert_to_tensor(s, dtype=tf.float32)
    a = tf.convert_to_tensor(a, dtype=tf.int32)
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    g_t, _ = discount_rewards(r.numpy(), dones.numpy(), gamma)
    g_t = tf.convert_to_tensor(g_t, dtype=tf.float32)

    # train critic
    with tf.GradientTape() as tape:
        critics = model_critic(s, training=True)
        critics = tf.squeeze(critics, axis=1)
        loss_critic = tf.keras.losses.mean_squared_error(g_t, critics)
    critic_grads = tape.gradient(loss_critic, model_critic.trainable_variables)
    opt_critic.apply_gradients(zip(critic_grads, model_critic.trainable_variables))

    # train actor
    with tf.GradientTape() as tape:
        logits = model_actor(s, training=True)
        log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a, logits=logits)
        advantages = g_t - tf.stop_gradient(tf.squeeze(model_critic(s, training=False), axis=1))
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        loss_actor = -tf.reduce_mean(log_prob * advantages)
    actor_grads = tape.gradient(loss_actor, model_actor.trainable_variables)
    opt_actor.apply_gradients(zip(actor_grads, model_actor.trainable_variables))

    return loss_critic.numpy(), loss_actor.numpy()

# hyperparams
GAMMA = 0.99
last_n_reward = 100
TRAIN_EPISODES = 2000
actor_lr = 3e-4
critic_lr = 1e-3
batch_size = 5000

# initialize models & optimizers
actor = actor_creator(obs_size, act_size)
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
    states, actions, rewards, n_dones, episode_reward = sample_traj(actor, batch=batch_size)
    critic_loss, actor_loss = train(actor, critic, actor_optimizer, critic_optimizer, states, actions, rewards, n_dones, GAMMA)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    episode_reward_history.append(episode_reward)
    episode_rewards.append(episode_reward)

    if len(episode_reward_history) > last_n_reward:
        del episode_reward_history[0]
    running_reward = np.mean(episode_reward_history)
    running_rewards.append(running_reward)

    pbar.set_postfix(EpisodeReward=f'{episode_reward:.2f}', RunningReward=f'{running_reward:.2f}')

    if running_reward >= 475:
        consistency += 1
        if consistency >= 10:
            print(f"Early stopping at episode {ep} since target achieved.")
            break
    else:
        consistency = 0

pbar.close()

# save only weights for visualization compatibility
actor.save_weights("vpg_actor_weights.h5")
critic.save_weights("vpg_critic_weights.h5")

# visualize training progress
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(running_rewards, label="running reward")
plt.plot(episode_rewards, label="episode reward", alpha=0.4)
plt.xlabel("episode")
plt.ylabel("reward")
plt.title("Training Results - Reward Evolution")
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
plt.savefig("training_progress.png")

# visualization of the trained actor using Pygame
actor.load_weights("vpg_actor_weights.h5")
env = gym.make("CartPole-v1", render_mode="human")  # separate env for testing
obs = env.reset()[0]

done = False
while not done:
    logit = actor(tf.convert_to_tensor(np.expand_dims(obs, axis=0), dtype=tf.float32), training=False)
    # deterministic action selection
    obs, _, terminated, truncated, _ = env.step(np.argmax(tf.nn.softmax(logit).numpy()[0]))
    done = terminated or truncated

env.close()
