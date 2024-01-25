# Q-learning Applied to OpenAI's Cart-Pole Problem

<img src=docs/learned.gif width=400px></img>

This is my implementation of Q-learning on OpenAI's cart pole problem. Despite the state space being continuous for this problem, I discretize the values into buckets allowing Q-learning to function. 

The model fairly consistently solves the environment in about 600 episodes.

##  


## Behvaiour

![Off-policy TD control algorithm (Q-learning)](docs/algorithm.png)

![Procedural algorithm](docs/procedural.png)

To allow the agent to explore the complex environment and build up an optimal policy without converging on a suboptimal policy too quickly we implement a decaying exploration and learning rate. The exploration rate decays as does the learning rate until after a large amount of exploration, the agent then exploits the more optimal path to balance the pole. 

The model has no separate exploration and exploitation phases, but rather gradually moves towards more exploitation as time goes on. This means that in some cases where it fails to explore the environment fully before it begins to exploit it, it can take significantly longer to solve the problem - however, this is quite rare. The model will always explore even in the later stages, just at a significantly reduced amount.

## Results

The learned model balancing the pole.

![Cart](docs/learned.gif)

## Runs

<img src="docs/Figure_1.png" width="400" />
<img src="docs/Figure_2.png" width="400" />
<img src="docs/Figure_3.png" width="400" />

## Inspired by / References
```bibtex
@book{ sutton_barto_2018, 
    title={Reinforcement Learning: An Introduction}, 
    publisher={MIT Press Ltd}, 
    author={Sutton, Richard S. and Barto, Andrew G.}, 
    year={2018}
}

@misc{ mnih_kavuk_silver_2015,
    title={Human-level control through deep reinforcement learning}
    url={https://www.nature.com/articles/nature14236},
    journal={Nature News},
    publisher={Nature Publishing Group},
    author={Volodymyr Mnih, Koray Kavukcuoglu, David Silver},
    year={2015}
}
```
