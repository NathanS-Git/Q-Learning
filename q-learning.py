import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
import numpy as np
from matplotlib import pyplot as plt
import os

def main():

    env = gym.make("CartPole-v1")

    actions = [0,1] # Action space. 0 - Pushes car to the left, 1 - Pushes cart to the right

    bc_x,bc_x_dot,bc_theta,bc_theta_dot = 2,1,6,12 # Number of bins per state value (Bin counts)
    
    # Creating bins
    x_bins = np.linspace(-4.8,4.8,bc_x-1)
    x_dot_bins = np.linspace(-0.5,0.5,bc_x_dot-1) # -0.5 to 0.5 was chosen simply through observing the cart. It doesn't seem to go beyond these values often.
    theta_bins = np.linspace(-0.418,0.418,bc_theta-1)
    theta_dot_bins = np.linspace(-1,1,bc_theta_dot-1) # Same story here. 
    bins = (x_bins,x_dot_bins,theta_bins,theta_dot_bins)

    observation,_ = env.reset()
    # Discretize the observation data
    state = tuple([np.digitize(observation[i],bins[i]) for i in range(len(observation))])

    gamma = 0.99

    alpha_start = 0.1 
    alpha_end = 0.1 
    alpha_end_episode = 200

    epsilon_start = 1
    epsilon_end = 0.1
    epsilon_end_episode = 200

    max_episode_count = 1000 # Max episode count

    # The Q-Table of values. e.g. Q[S][A]
    Q_table = {}

    # Initialize all Q(s,a) to 0
    for x in range(bc_x):
        for x_dot in range(bc_x_dot):
            for theta in range(bc_theta):
                for theta_dot in range(bc_theta_dot):
                    Q_table[(x,x_dot,theta,theta_dot)] = {}
                    for a in actions:
                        Q_table[(x,x_dot,theta,theta_dot)][a] = 0

    previous_total_rewards = [] # History of reward per episode

    for episode in range(max_episode_count):
        #alpha = max(min_alpha,alpha/(1+alpha_decay*episode))
        alpha = max(episode*((alpha_end-alpha_start)/alpha_end_episode)+alpha_start, alpha_end)
        #epsilon = max(min_epsilon,epsilon/(1+epsilon_decay*episode))
        epsilon = max(episode*((epsilon_end-epsilon_start)/epsilon_end_episode)+epsilon_start, epsilon_end)

        observation,_ = env.reset() # x x_dot theta theta_dot (Initialize S)
        # Discretize the observation data
        state = tuple([np.digitize(observation[i],bins[i]) for i in range(len(observation))])

        done = False
        truncated = False
        total_reward = 0
        while not done and not truncated:

            # Choose A from S using policy derived from Q (e.g. epsilon-greedy)
            if np.random.random() < epsilon:
                action = np.random.choice(actions)
            else:
                action = max(Q_table[state],key=Q_table[state].get)

            # Take action A, observe R, S'
            next_observation,reward,done,truncated,_ = env.step(action)
            
            total_reward += reward

            # Discretize the observation data
            next_state = tuple([np.digitize(next_observation[i],bins[i]) for i in range(len(next_observation))])

            # Q(S,A) <- Q(S,A) + alpha*( R + gamma*maxQ(S',a) - Q(S,A) )
            Q_table[state][action] = Q_table[state][action]+alpha*(reward + gamma*max(Q_table[next_state].values()) - Q_table[state][action])
            state = next_state # S <- S'


        previous_total_rewards.append(total_reward)

        if len(previous_total_rewards) > 100:
            print(sum(previous_total_rewards[-100:])/100,epsilon,episode,total_reward)

    #plt.title("Q-Learning")
    #plt.xlabel("Episode")
    #plt.ylabel("Reward")
    #plt.plot(previous_total_rewards)
    #plt.show()



if (__name__ == "__main__"):
    main()
