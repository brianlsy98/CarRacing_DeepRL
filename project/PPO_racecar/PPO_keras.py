#!/usr/bin/env python2

from __future__ import print_function
from imp import new_module


import sys
import os
import rospkg
import rospy
from sim2real.msg import Result, Query

PATH = rospkg.RosPack().get_path("sim2real") + "/scripts"
print(PATH)
sys.path.append(PATH)

import numpy as np
import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from joblib import load

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import gym
import scipy.signal
import time

project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval1.yaml"

TEAM_NAME = "GOTOMARS"
# Code by Sungyoung Lee
team_path = project_path + "/project/IS_" + TEAM_NAME





def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage .
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:

    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize he advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer)
        )
        self.advantage_buffer = (self.advantage_buffer -advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)


class PPO_keras():

    def __init__(self):
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load("eval1")

        self.maxVel = 3.0
        self.minVel = 1.5
        self.curVel = self.minVel
        self.maxAng = 1.5

        self.time_limit = 100.0
        # Hyperparameters of the PPO algorithm
        self.steps_per_epoch = 4000
        self.epochs = 30
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.policy_learning_rate = 3e-4
        self.value_function_learning_rate = 1e-3
        self.hidden_sizes = (64, 64, 32)

        self.train_policy_iterations = 80
        self.train_value_iterations = 80
        self.lam = 0.97
        self.target_kl = 0.01

        # True if you want to render the environment
        self.render = True


        self.observation_dimensions = 26 + 2
        self.num_actions = 7

        self.buffer = Buffer(self.observation_dimensions, self.steps_per_epoch)

        self.observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        self.logits = mlp(self.observation_input, list(self.hidden_sizes) + [self.num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=self.observation_input, outputs=self.logits)
        self.value = tf.squeeze(
            mlp(self.observation_input, list(self.hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=self.observation_input, outputs=self.value)

        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)

    def load_actor(self, actor):
        self.actor = actor

    def load_critic(self, critic):
        self.critic = critic

    def set_layers(self, size):
        self.hidden_sizes = size

    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability


    # Sample action from actor
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        print(action)
        print(logits)
        return logits, action


    # Train the policy by maximizing the PPO-Clip objective
    @tf.function
    def train_policy(self,
        observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape: # Record operations for automatic differentiation
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))



    def action_to_angle_and_vel(self, action, prv_steer, prv_vel):       
        
        if action == 0:      # ang += 0.1  vel += 0.1
            input_steering = prv_steer + 0.1
            input_velocity = prv_vel + 0.1

        elif action == 1:    # ang += 0.1  vel const
            input_steering = prv_steer + 0.1
            input_velocity = prv_vel

        elif action == 2:    # ang const   vel += 0.1
            input_steering = prv_steer
            input_velocity = prv_vel + 0.1
            
        elif action == 3:    # ang const   vel const
            input_steering = prv_steer
            input_velocity = prv_vel

        elif action == 4:    # ang const   vel -= 0.1
            input_steering = prv_steer
            input_velocity = prv_vel - 0.1

        elif action == 5:    # ang -= 0.1  vel -= 0.1
            input_steering = prv_steer - 0.1
            input_velocity = prv_vel - 0.1

        elif action == 6:    # ang -= 0.1  vel const
            input_steering = prv_steer - 0.1
            input_velocity = prv_vel

        input_steering = np.clip(input_steering, -self.maxAng, self.maxAng)
        input_velocity = np.clip(input_velocity, self.minVel, self.maxVel)

        return input_steering, input_velocity
        


    def obs_dim_change(self, obs_high):   # from 1083 -> 28
        obs_prv_action = []
        obs_lidar = []
        obs_low = []
        obs_high[obs_high == float('inf')] = 30
        for i in range(len(obs_high[0])-2):
            obs_lidar.append(obs_high[0][i])
        obs_prv_action.extend([obs_high[0][-2], obs_high[0][-1]])
        for i in range(26):
            obs_low.append(np.mean(obs_lidar[4*i:4*i+80]))
        obs_low.extend(obs_prv_action)
        return obs_low


    def get_action_from_expert(self, obs):

        obs[obs == float('inf')] = 30
        obs_clip = np.array(obs[:,:-2])
        s = self.normalized_obs.transform(obs_clip)
        [[angle_act, linvel_act]] = self.gp.predict(s)
        i_steering, i_linear = self.normalized_act.inverse_transform([angle_act, linvel_act])

        return [[i_steering, i_linear]]


    def pretrain_withBC(self, track_name):

        ###### Load Expert DATA ######
        try:
            self.kernel = C(constant_value=1.0, constant_value_bounds=(1e-5, 10.0))* RBF(length_scale=1.0, length_scale_bounds=(1e-5, 10.0)) + C(constant_value = 0.5, constant_value_bounds = (1e-5, 1.0)) * RBF(length_scale = 1.0, length_scale_bounds = (1e-5, 100.0))
            self.gp = GaussianProcessRegressor(kernel = self.kernel, alpha = 0.2)
            self.gp_file = team_path + "/demo.joblib"
            self.obs_file = team_path + "/obs_file.pkl"
            self.act_file = team_path + "/act_file.pkl"
            self.gp = load(self.gp_file)
            self.normalized_obs = load(self.obs_file)
            self.normalized_act = load(self.act_file)
            print("expert model loaded for BC!!")
        except:
            print("loading error for BC..")
        ##############################    

        

        print("pretraining START for "+track_name)
        
        observation, episode_return, episode_length = self.env.reset(name = track_name), 0, 0
        observation = np.reshape(observation, [1,-1])

        for epoch in range(self.epochs//10):
            sum_return = 0
            sum_length = 0
            num_episodes = 0
            
            for t in range(self.steps_per_epoch):
                
                act = self.get_action_from_expert(observation)
                input_steering = np.clip(act[0][0], -self.maxAng, self.maxAng)
                input_velocity = np.clip(act[0][1], self.minVel, self.maxVel)

                observation, reward, done, logs = self.env.step([input_steering, input_velocity])
                observation = np.reshape(observation, [1, -1])

                # print("")
                print(input_steering); print(input_velocity)
                # print("")
                # print(observation); print(len(observation[0]))
                print(observation[0][-1]); print(observation[0][-2])
                # print(observation_new[0][-1]); print(observation_new[0][-2])
                # ang_acc = observation_new[0][-1] - observation[0][-1]
                # lin_acc = observation_new[0][-2] - observation[0][-2]
                # print(ang_acc); print(lin_acc)
        
        return

    


    def train(self, track_name):

        print("training START for "+track_name)

        observation, episode_return, episode_length = self.env.reset(name = track_name), 0, 0

        rrecord = []
        for epoch in range(self.epochs):
            sum_return = 0
            sum_length = 0
            num_episodes = 0
            
            prev_steer = 0.0; prev_vel = self.minVel

            for t in range(self.steps_per_epoch):
                if self.render:
                    self.env.render()
                observation = observation.reshape(1, -1)
                obs_dim28 = self.obs_dim_change(observation)
                
                # logits for critic update
                logits, action = self.sample_action(np.array(obs_dim28).reshape(1, -1))
                
                input_steering, input_velocity = self.action_to_angle_and_vel(action, prev_steer, prev_vel)
                observation_new, reward, done, logs = self.env.step([input_steering, input_velocity])
                prev_steer = input_steering; prev_vel = input_velocity

                if np.min(observation_new[:-2]) < 0.6: # wall very close in front 
                        done = True
                reward += input_velocity - self.maxVel
                if done and logs['info'] != 3: reward -= 500

                episode_return += reward
                episode_length += 1


                value_t = self.critic(np.array(obs_dim28).reshape(1, -1))
                logprobability_t = self.logprobabilities(logits, action)
                self.buffer.store(obs_dim28, action, reward, value_t, logprobability_t)
                observation = observation_new

                

                terminal = done
                if terminal or (t == self.steps_per_epoch - 1):
                    last_value = 0 if done else self.critic(np.array(obs_dim28).reshape(1,-1))
                    self.buffer.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = self.env.reset(name = track_name), 0, 0

            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()


            for _ in range(self.train_policy_iterations):
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    break

            for _ in range(self.train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)


            print(
                " Epoch: "+str(epoch + 1)+". Mean Return: "+str(sum_return / num_episodes)+". Mean Length: "+str(sum_length / num_episodes)    
            )
            rrecord.append(sum_return / num_episodes)

            if epoch % 3 == 2:
                Actor_for_save = self.actor
                Critic_for_save = self.critic
                Actor_for_save.save_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/PPO/'+str(track_name)+'/PPO_actor_weights_'+str(epoch//3))
                Critic_for_save.save_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/PPO/'+str(track_name)+'/PPO_critic_weights_'+str(epoch//3))


            if epoch == self.epochs-1:
                # plot [episode, reward] history
                x = [i+1 for i in range(len(rrecord))]
                plt.plot(x, rrecord)
                plt.title('episode rewards')
                plt.xlabel('episodes')
                plt.ylabel('rewards')
                plt.show()



if __name__ == '__main__':
    PPO_keras().train()