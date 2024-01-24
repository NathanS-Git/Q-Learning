import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque, namedtuple

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
            nn.Linear(env.observation_space.shape[0], 50),
            nn.GELU(),
            nn.Linear(50,25),
            nn.GELU(),
            nn.Linear(25, env.action_space.n)
        )

    def forward(self, x: torch.Tensor):
        return self.network(x)


def main():
    env = gym.make('CartPole-v1')

    replay = MemoryReplay(5000)

    q_network = DQN(env).to(device)
    target_network = DQN(env).to(device)
    target_network.load_state_dict(q_network.state_dict())
    #target_network = copy.deepcopy(q_network)

    optimizer = optim.AdamW(q_network.parameters(), amsgrad=True)

    epsilon = 0.1
    gamma = 0.1
    C = 500

    batch_size = 256

    episode = 0
    max_episode_count = 1000

    previous_total_rewards = []

    while episode < max_episode_count:
        state,_ = env.reset() 
        
        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(q_network(torch.tensor(state, device=device))).item()

            next_state,reward,done,truncated,_ = env.step(action)

            total_reward += reward

            next_state = None if done is True else next_state

            replay.save(state, [action], [reward], next_state)

            next_state = state

            # Optimize the model
            if len(replay) > batch_size:
                
                transitions = replay.sample(batch_size)
                state_batch,action_batch,reward_batch,next_state_batch = list(zip(*transitions))

                q_values = q_network(torch.tensor(state_batch, device=device)).gather(1, torch.tensor(action_batch, device=device))

                mask = list(map(lambda x: x is not None, next_state_batch))
                non_terminated_next_states = tuple(next_state for next_state in next_state_batch if next_state is not None)
                target_q_values = target_network(torch.tensor(non_terminated_next_states, device=device)).max(1,keepdim=True).values
                y = torch.tensor(reward_batch, device=device)

                #print(f"MASK: {mask} Y:{y} TARGET:{target_q_values}")
                y[mask] += target_q_values * gamma
                
                optimizer.zero_grad()
                loss = nn.HuberLoss()(q_values, y)
                loss.backward()
                optimizer.step()
        
        if episode % C == 0:
            target_network.load_state_dict(q_network.state_dict())


        previous_total_rewards.append(total_reward)

        if len(previous_total_rewards) > 100:
            print(sum(previous_total_rewards[-100:])/100)


if (__name__ == "__main__"):
    main()
