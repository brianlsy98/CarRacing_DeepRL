from importlib.machinery import ModuleSpec
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
from utils import plot_learning_curve
import rospkg

project_path = rospkg.RosPack().get_path("sim2real")
team_path = project_path + "/project/IS_GOTOMARS"

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self, path):
        T.save(self.state_dict(), path+"actor_torch_ppo")

    def load_checkpoint(self, modelpath):
        self.load_state_dict(T.load(modelpath))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self, path):
        T.save(self.state_dict(), path+"critic_torch_ppo")

    def load_checkpoint(self, modelpath):
        self.load_state_dict(T.load(modelpath))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10, path = team_path):
        
        self.model_path = path+"/project/PPO/PPOmodels/"

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint(self.model_path)
        self.critic.save_checkpoint(self.model_path)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint(self.model_path)
        self.critic.load_checkpoint(self.model_path)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value


    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()               



    def fine_control(self, fc):

        if fc == 0: ang = -1.5
        elif fc == 1: ang = -1.2
        elif fc == 2: ang = -0.9
        elif fc == 3: ang = -0.6
        elif fc == 4: ang = -0.3
        elif fc == 5: ang = 0.0
        elif fc == 6: ang = 0.3
        elif fc == 7: ang = 0.6
        elif fc == 8: ang = 0.9
        elif fc == 9: ang = 1.2
        elif fc == 10: ang = 1.5

        return ang

    def obs_normalize(self, obs_high):   # from 1083 -> 26 + 2
        obs_prv_action = []
        obs_lidar = []
        
        for i in range(26):
            obs_lidar.append(np.mean(obs_high[40*i:40*i+81]))
        obs_prv_action.extend([obs_high[1082], obs_high[1081]])
        
        # normalize -> element sum to 1
        obs_low_lidar = np.array(obs_lidar)/np.sum(obs_lidar)
        obs_low_prvact = np.array(obs_prv_action)
        return np.concatenate((obs_low_lidar, obs_low_prvact)), obs_low_lidar, obs_low_prvact


    def train(self, tracklist, e):

        env = e
        N = 20
        
        n_games = 1000

        figure_file = self.model_path+'cartpole.png'

        best_score = env.reward_range[0]
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0

        for i in range(n_games):
            # Get initial state
            # ====== Initialization  ====== #
            world_name = random.choice(np.array(tracklist))
            print(tracklist)
            print(world_name)
            obs = env.reset(name = world_name)
            obs = np.reshape(obs, [1,-1])
            obs[obs == float('inf')] = 10

            # ** for PPO model ** #
            obs_norm, _, _ = self.obs_normalize(obs[0])
            # ******************* #
            # ============================= #


            done = False
            score = 0
            while not done:
                action, prob, val = self.choose_action(obs_norm)

                # == Controls from PPO == #
                input_steering = self.fine_control(action)
                input_velocity = (1.5-abs(input_steering))+1.5
                # ======================= #
                # obs_, reward, done, info = env.step([input_steering, input_velocity])
                obs_, reward, done, info = env.step([input_steering, input_velocity])
                obs_ = np.reshape(obs_, [1,-1])
                obs_[obs_ == float('inf')] = 10

                if (env.get_pose().position.z) > 0.001 : done = True
                if done == True and info['info'] != 3 : reward -= 500

                obs_norm_, _, _ = self.obs_normalize(obs_[0])

                n_steps += 1
                score += reward
                self.remember(obs_norm, action, prob, val, reward, done)
                if n_steps % N == 0:
                    self.learn()
                    learn_iters += 1
                obs_norm = obs_norm_

            env.reset(name = world_name)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                self.save_models()
                
            print("")
            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', n_steps, 'learning_steps', learn_iters)
            
            x = [j+1 for j in range(i+1)]
            if len(x) % 100 == 0:
                plot_learning_curve(x, score_history, figure_file)

