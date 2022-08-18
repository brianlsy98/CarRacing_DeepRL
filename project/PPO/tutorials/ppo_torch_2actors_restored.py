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
        self.probs1 = []
        self.probs2 = []
        self.vals = []
        self.actions1 = []
        self.actions2 = []
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
                np.array(self.actions1),\
                np.array(self.actions2),\
                np.array(self.probs1),\
                np.array(self.probs2),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action1, action2, probs1, probs2, vals, reward, done):
        self.states.append(state)
        self.actions1.append(action1)
        self.actions2.append(action2)
        self.probs1.append(probs1)
        self.probs2.append(probs2)
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
        
        self.model_path = path+"/project/PPO/PPOmodels_2actors/"

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor1 = ActorNetwork(n_actions, input_dims, alpha)   # steering
        self.actor2 = ActorNetwork(n_actions, input_dims, alpha)   # velocity
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action1, action2, probs1, probs2, vals, reward, done):
        self.memory.store_memory(state, action1, action2, probs1, probs2, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor1.save_checkpoint(self.model_path+"1st_")
        self.actor2.save_checkpoint(self.model_path+"2nd_")
        self.critic.save_checkpoint(self.model_path)

    def load_models(self):
        print('... loading models ...')
        self.actor1.load_checkpoint(self.model_path)
        self.actor2.load_checkpoint(self.model_path)
        self.critic.load_checkpoint(self.model_path)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor1.device)

        dist1 = self.actor1(state)
        action1 = dist1.sample()

        dist2 = self.actor2(state)
        action2 = dist2.sample()

        value = self.critic(state)


        probs1 = T.squeeze(dist1.log_prob(action1)).item()
        probs2 = T.squeeze(dist2.log_prob(action2)).item()
        action1 = T.squeeze(action1).item()
        action2 = T.squeeze(action2).item()
        value = T.squeeze(value).item()

        return action1, action2, probs1, probs2, value


    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr1, action_arr2, old_prob_arr1, old_prob_arr2, vals_arr,\
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
            advantage = T.tensor(advantage).to(self.actor1.device)

            values = T.tensor(values).to(self.actor1.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor1.device)
                old_probs1 = T.tensor(old_prob_arr1[batch]).to(self.actor1.device)
                actions1 = T.tensor(action_arr1[batch]).to(self.actor1.device)
                old_probs2 = T.tensor(old_prob_arr2[batch]).to(self.actor2.device)
                actions2 = T.tensor(action_arr2[batch]).to(self.actor2.device)

                dist1 = self.actor1(states)
                dist2 = self.actor2(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs1 = dist1.log_prob(actions1)
                prob_ratio1 = new_probs1.exp() / old_probs1.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs1 = advantage[batch] * prob_ratio1
                weighted_clipped_probs1 = T.clamp(prob_ratio1, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor1_loss = -T.min(weighted_probs1, weighted_clipped_probs1).mean()

                new_probs2 = dist2.log_prob(actions2)
                prob_ratio2 = new_probs2.exp() / old_probs2.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs2 = advantage[batch] * prob_ratio2
                weighted_clipped_probs2 = T.clamp(prob_ratio1, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor2_loss = -T.min(weighted_probs2, weighted_clipped_probs2).mean()


                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor1_loss + actor2_loss + 0.5*critic_loss
                self.actor1.optimizer.zero_grad()
                self.actor2.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor1.optimizer.step()
                self.actor2.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()               



    def steering_control(self, fc):

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

    def velocity_control(self, fc):

        if fc == 0: vel = 1.5
        elif fc == 1: vel = 1.6
        elif fc == 2: vel = 1.7
        elif fc == 3: vel = 1.9
        elif fc == 4: vel = 2.1
        elif fc == 5: vel = 2.3
        elif fc == 6: vel = 2.5
        elif fc == 7: vel = 2.7
        elif fc == 8: vel = 2.8
        elif fc == 9: vel = 2.9
        elif fc == 10: vel = 3.0

        return vel

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
                action1, action2, prob1, prob2, val = self.choose_action(obs_norm)

                # == Controls from PPO == #
                input_steering = self.steering_control(action1)
                input_velocity = self.velocity_control(action2)
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
                self.remember(obs_norm, action1, action2, prob1, prob2, val, reward, done)
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

