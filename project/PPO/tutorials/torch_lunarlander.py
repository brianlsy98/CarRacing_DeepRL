# PPO torch tutorial for cartpole


from pathlib import Path

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from PPO.replay import Episode, History



import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np

from torch.utils.data import Dataset






################################
# ********** REPLAY ********** #
################################
def cumulative_sum(array, gamma=1.0):
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


class Episode:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(
        self, observation, action, reward, value, log_probability, reward_scale=20
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)

        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]


def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()


class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build_dataset(self):
        for episode in self.episodes:
            self.observations += episode.observations
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

        assert (
            len(
                {
                    len(self.observations),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probabilities),
                }
            )
            == 1
        )

        self.advantages = normalize_list(self.advantages)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )
################################







################################
# ********** MODELs ********** #
################################
class PolicyNetwork(torch.nn.Module):
    def __init__(self, n=4, in_dim=128):
        super(PolicyNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)

        self.fc4 = torch.nn.Linear(128, n)

        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = self.fc4(x)

        y = F.softmax(y, dim=-1)

        return y

    def sample_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        dist = Categorical(y)

        action = dist.sample()

        log_probability = dist.log_prob(action)

        return action.item(), log_probability.item()

    def best_action(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state).squeeze()

        action = torch.argmax(y)

        return action.item()

    def evaluate_actions(self, states, actions):
        y = self(states)

        dist = Categorical(y)

        entropy = dist.entropy()

        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy


class ValueNetwork(torch.nn.Module):
    def __init__(self, in_dim=128):
        super(ValueNetwork, self).__init__()

        self.fc1 = torch.nn.Linear(in_dim, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)

        self.fc4 = torch.nn.Linear(128, 1)

        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = self.fc4(x)

        return y.squeeze(1)

    def state_value(self, state):

        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        return y.item()


def train_value_network(value_model, value_optimizer, data_loader, epochs=4):
    epochs_losses = []

    for i in range(epochs):

        losses = []

        for observations, _, _, _, rewards_to_go in data_loader:
            observations = observations.float().to(device)
            rewards_to_go = rewards_to_go.float().to(device)

            value_optimizer.zero_grad()

            values = value_model(observations)

            loss = F.mse_loss(values, rewards_to_go)

            loss.backward()

            value_optimizer.step()

            losses.append(loss.item())

        mean_loss = np.mean(losses)

        epochs_losses.append(mean_loss)

    return epochs_losses


def ac_loss(new_log_probabilities, old_log_probabilities, advantages, epsilon_clip=0.2):
    probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
    clipped_probabiliy_ratios = torch.clamp(
        probability_ratios, 1 - epsilon_clip, 1 + epsilon_clip
    )

    surrogate_1 = probability_ratios * advantages
    surrogate_2 = clipped_probabiliy_ratios * advantages

    return -torch.min(surrogate_1, surrogate_2)


def train_policy_network(
    policy_model, policy_optimizer, data_loader, epochs=4, clip=0.2
):
    epochs_losses = []

    c1 = 0.01

    for i in range(epochs):

        losses = []

        for observations, actions, advantages, log_probabilities, _ in data_loader:
            observations = observations.float().to(device)
            actions = actions.long().to(device)
            advantages = advantages.float().to(device)
            old_log_probabilities = log_probabilities.float().to(device)

            policy_optimizer.zero_grad()

            new_log_probabilities, entropy = policy_model.evaluate_actions(
                observations, actions
            )

            loss = (
                ac_loss(
                    new_log_probabilities,
                    old_log_probabilities,
                    advantages,
                    epsilon_clip=clip,
                ).mean()
                - c1 * entropy.mean()
            )

            loss.backward()

            policy_optimizer.step()

            losses.append(loss.item())

        mean_loss = np.mean(losses)

        epochs_losses.append(mean_loss)

    return epochs_losses
################################






################################
# ******** MAIN FILEs ******** #
################################
def main(
    env_name="LunarLander-v2",
    reward_scale=20.0,
    clip=0.2,
    log_dir="../logs",
    learning_rate=0.001,
    state_scale=1.0,
):
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=env_name, comment=env_name)

    env = gym.make(env_name)
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

    n_epoch = 4

    max_episodes = 20
    max_timesteps = 400

    batch_size = 32

    max_iterations = 200

    history = History()

    epoch_ite = 0
    episode_ite = 0

    for ite in tqdm(range(max_iterations)):

        if ite % 50 == 0:
            torch.save(
                policy_model.state_dict(),
                Path(log_dir) / (env_name + f"_{str(ite)}_policy.pth"),
            )
            torch.save(
                value_model.state_dict(),
                Path(log_dir) / (env_name + f"_{str(ite)}_value.pth"),
            )

        for episode_i in range(max_episodes):

            observation = env.reset()
            episode = Episode()

            for timestep in range(max_timesteps):

                action, log_probability = policy_model.sample_action(
                    observation / state_scale
                )
                value = value_model.state_value(observation / state_scale)

                new_observation, reward, done, info = env.step(action)

                episode.append(
                    observation=observation / state_scale,
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=reward_scale,
                )

                observation = new_observation

                if done:
                    episode.end_episode(last_value=0)
                    break

                if timestep == max_timesteps - 1:
                    value = value_model.state_value(observation / state_scale)
                    episode.end_episode(last_value=value)

            episode_ite += 1
            writer.add_scalar(
                "Average Episode Reward",
                reward_scale * np.sum(episode.rewards),
                episode_ite,
            )
            writer.add_scalar(
                "Average Probabilities",
                np.exp(np.mean(episode.log_probabilities)),
                episode_ite,
            )

            history.add_episode(episode)

        history.build_dataset()
        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)

        policy_loss = train_policy_network(
            policy_model, policy_optimizer, data_loader, epochs=n_epoch, clip=clip
        )

        value_loss = train_value_network(
            value_model, value_optimizer, data_loader, epochs=n_epoch
        )

        for p_l, v_l in zip(policy_loss, value_loss):
            epoch_ite += 1
            writer.add_scalar("Policy Loss", p_l, epoch_ite)
            writer.add_scalar("Value Loss", v_l, epoch_ite)

        history.free_memory()


if __name__ == "__main__":

    main(
        reward_scale=20.0,
        clip=0.2,
        env_name="LunarLander-v2",
        learning_rate=0.001,
        state_scale=1.0,
        log_dir="logs/Lunar"
    )
################################