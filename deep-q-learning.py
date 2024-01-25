import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class MemoryReplay():
    def __init__(self, capacity):
        self._memory = deque(maxlen=capacity)

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def save(self, state, action, reward, next_state):
        self._memory.append(Transition(state,action,reward,next_state))

    def __len__(self):
        return len(self._memory)



class DQN(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.GELU(),
            nn.Linear(128,128),
            nn.GELU(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


def main():
    env = gym.make("CartPole-v1")

    q_network = DQN(env).to(device)
    target_network = DQN(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    #target_network = copy.deepcopy(q_network)

    epsilon_start = 1
    epsilon_end = 0.01
    epsilon_end_episode = 500

    lr = 0.00015
    gamma = 0.99
    C = 13

    batch_size = 256

    max_episode_count = 1000
    replay_bank_size = 5000

    optimizer = optim.AdamW(q_network.parameters(), lr=lr, amsgrad=True)
    replay = MemoryReplay(replay_bank_size)

    previous_total_rewards = []

    for episode in range(max_episode_count):
        state,_ = env.reset()
        
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
            epsilon = max(episode*((epsilon_end-epsilon_start)/epsilon_end_episode)+epsilon_start, epsilon_end)

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(torch.tensor(state, device=device))).item()

            next_state,reward,done,truncated,_ = env.step(action)

            total_reward += reward

            next_state = None if done else next_state

            replay.save(state, [action], [reward], next_state)

            state = next_state

            # Optimize the model
            #if len(replay) > batch_size:
            transitions = replay.sample(min(batch_size,len(replay)))
            state_batch,action_batch,reward_batch,next_state_batch = list(zip(*transitions))

            q_values = q_network(torch.tensor(numpy.array(state_batch), device=device)).gather(1, torch.tensor(action_batch, device=device))

            mask = list(map(lambda x: x is not None, next_state_batch))
            non_terminated_next_states = tuple(next_state for next_state in next_state_batch if next_state is not None)
            with torch.no_grad():
                target_q_values = target_network(torch.tensor(numpy.array(non_terminated_next_states), device=device)).max(1,keepdim=True).values

            y = torch.tensor(reward_batch, device=device, dtype=torch.float32)

            y[mask] += target_q_values * gamma
            
            optimizer.zero_grad()
            loss = torch.nn.HuberLoss()(q_values, y)
            loss.backward()
            optimizer.step()

        
        if episode % C == 0:
            target_network.load_state_dict(q_network.state_dict())


        previous_total_rewards.append(total_reward)

        if len(previous_total_rewards) > 100:
            print(sum(previous_total_rewards[-100:])/100,epsilon,episode,total_reward,len(replay))


if (__name__ == "__main__"):
    main()
