import os
import sys
import tensorflow as tf
import numpy as np
import cv2 as cv
import collections

from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces
import pybullet as p

import matplotlib
import matplotlib.pyplot as plt


plt.ion()


def DQN(num_actions, stack):
    # Network defined by the Deepmind paper
    inputs = tf.keras.layers.Input(shape=(84, 84, stack))

    # Convolutions on the frames on the screen
    layer1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = tf.keras.layers.Flatten()(layer3)

    layer5 = tf.keras.layers.Dense(512, activation="relu")(layer4)
    action = tf.keras.layers.Dense(num_actions, activation="linear")(layer5)

    return tf.keras.Model(inputs=inputs, outputs=action)



def main():
    # Configuration paramaters for the whole setup
    seed = 42
    stack = 4
    gamma = 0.99  # Discount factor for past rewards
    epsilon = 1.0  # Epsilon greedy parameter
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1.0  # Maximum epsilon greedy parameter
    epsilon_interval = (
        epsilon_max - epsilon_min
    )  # Rate at which to reduce chance of random action being taken
    batch_size = 32  # Size of batch taken from replay buffer
    max_steps_per_episode = 10000
    num_actions=7
    checkpoint_path = "qlearning_models/cp-{episode:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model = DQN(num_actions, stack)
    model_target = DQN(num_actions, stack)
    # In the Deepmind paper they use RMSProp however then Adam optimizer
    # improves training time
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    env = KukaDiverseObjectEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20)

    env.render('human')

    # Experience replay buffers
    action_history = []
    state_history = []
    state_next_history = []
    rewards_history = []
    done_history = []
    episode_reward_history = []
    running_rewards = []
    running_reward = 0
    episode_count = 0
    frame_count = 0
    # Number of frames to take random action and observe output
    epsilon_random_frames = 50000
    # Number of frames for exploration
    #epsilon_greedy_frames = 1000000.0
    epsilon_greedy_frames = 10000.0
    # Maximum replay length
    # Note: The Deepmind paper suggests 1000000 however this causes memory issues
    max_memory_length = 1000000
    # Train the model after 4 actions
    update_after_actions = 4
    # How often to update the target network
    #update_target_network = 10000
    update_target_network = 1000
    # Using huber loss for stability
    loss_function = tf.keras.losses.Huber()

    #print(f"Saving model in episode: {episode_count}")
    #model.save_weights(checkpoint_path.format(episode=episode_count))

    #model.save_weights(checkpoint_path.format(reward=0))

    while True:  # Run until solved
        state = np.array(env.reset())
        state = process_state(state)
        episode_reward = 0
        
        stacked_states = collections.deque(stack*[state],maxlen=stack)
        #print(f" stacked_states{stacked_states.shape}")
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.
            env.render('human')
            frame_count += 1
    
            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                states_stack = np.array(stacked_states)
                states_stack = np.moveaxis(states_stack, 0, -1)
                state_tensor = tf.convert_to_tensor(states_stack)
                #print(f"state tensor shape : {tf.shape(state_tensor).numpy()}")
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
    
            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
    
            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            
            state_next = np.array(state_next)
            state_next = process_state(state_next)

            if not done:
                next_stacked_states = stacked_states
                next_stacked_states.append(state_next)
                #print(f"state_next: {state_next.shape}\nstacked: {np.moveaxis(np.array(next_stacked_states), 0, -1).shape}")
    
            episode_reward += reward
    
            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(stacked_states)
            state_next_history.append(next_stacked_states)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next
    
            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
    
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history)), size=batch_size)
    
                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices])
                state_sample = np.moveaxis(state_sample, 1, -1)
                state_next_sample = np.array([state_next_history[i] for i in indices])
                state_next_sample = np.moveaxis(state_next_sample, 1, -1)

                #print(f"next state: {state_next_sample.shape}")

                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )
                #print(f"state tensor shape : {tf.shape(tf.convert_to_tensor(state_next_sample).numpy()}")
                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = model_target.predict(state_next_sample)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )
    
                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample
    
                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)
    
                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)
    
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)
    
                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))
                print(f"Saving model in episode: {episode_count}")
                model.save_weights(checkpoint_path.format(episode=episode_count))
    
            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]
    
            if done:
                break
    
        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)
        running_rewards.append(running_reward)
        plot_rewards(running_rewards)
    
        episode_count += 1
    
        if running_reward > 0.5:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            print(f"Saving model with reward: {episode_reward}")
            model.save_weights(checkpoint_path.format(episode=episode_count))
            break

def get_screen(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env._get_observation()#.transpose((2, 0, 1))
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Resize, and add a batch dimension (BCHW)
    return screen

def plot_rewards(rewards):
    plt.plot(rewards)
    plt.draw()
    plt.pause(0.0001)
    plt.clf()

def process_state(screen):
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    screen = cv.resize(screen, (84,84))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Resize, and add a batch dimension (BCHW)
    return screen

def plot_screen(screen):
    plt.figure()
    plt.imshow(screen,cmap='Greys',
           interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


def test_env():
    env = KukaDiverseObjectEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)
    for i in range(2):
        state = np.array(env.reset())
        #the state is the screen
        print(f"State: {state.shape}")
        plot_screen(get_screen(env))


#test_env()
main()