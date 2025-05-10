import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

dt = 0.01           # time step, set to the same as control methods simulations in MATLAB
T = 5.0             # total sim time, same

def actor_creator(obs_size, act_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(obs_size,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(act_size)])
    return model

env = gym.make("CartPole-v1", render_mode="human")  # separate env with render
actor = actor_creator(4, 2)
actor.load_weights("vpg_actor_weights.h5")
obs, _ = env.reset()
obs[2] = np.deg2rad(5.0)    # initial tilt, same
env.state = obs             # update environment's state

t = [0.0]
hist  = [obs.copy()]
for i in range(int(T / dt)):
    logits = actor(tf.convert_to_tensor(obs[None], tf.float32), training=False)
    action = np.argmax(tf.nn.softmax(logits)[0].numpy())
    obs, _, done, _, _ = env.step(action)
    hist.append(obs.copy())
    t.append((i + 1) * dt)
    if done:
        break

env.close()

arr = np.array(hist)    # convert to array for compatibility & efficiency
plt.subplot(3, 1, 1)    # θ
plt.plot(t, np.rad2deg(arr[:, 2]))
plt.ylabel('θ (deg)')
plt.grid(True)

plt.subplot(3, 1, 2)    # x
plt.plot(t, arr[:, 0])
plt.ylabel('x (m)')
plt.grid(True)

plt.subplot(3, 1, 3)    # v
plt.plot(t, arr[:, 1])
plt.xlabel('time (s)')
plt.ylabel('v (m/s)')
plt.grid(True)

plt.tight_layout()
plt.savefig('cartpole_response.png')
plt.show()