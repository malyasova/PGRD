Implementation of the first experiment from the [Reward Design via Online Gradient Ascent](https://proceedings.neurips.cc/paper/2010/file/168908dd3227b8358eababa07fcaf091-Paper.pdf) paper.

The paper considers the scenario of an agent designer training an autonomous agent using Reinforcement Learning. 
The designer could use his own reward function to train the agent. However, a modified reward can accelerate learning and improve performance of a bounded agent.
The paper sets out to find the optimal reward function via gradient ascent for a family of bounded agents that behave 
according to repeated local model-based planning. The conclustion is that the Policy Gradient for Reward Design (PGRD) algorithm
can improve reward functions in agents with computational  limitations necessitating small bounds on the depth of planning.

PGRD builds on the insight that the agent's planning algorithm takes a reward as input and outputs behavior.
The true reward produced by this behavior can be differentiated with respect to the learning reward.
This way, the optimal learning reward can be found via gradient ascent.

*Experiment*

This repository reproduces the first experiment from the paper. The aim of the experiment is to demonstrate that PGRD can improve the reward and accelerate learning.
It is set in the foraging environment.
"The foraging environment is a 3×3 grid world with 3 dead-end corridors (rows) separated by impassable walls. The agent (bird) has
four available actions corresponding to each cardinal direction. Movement in the intended direction
fails with probability 0.1, resulting in movement in a random direction. If the resulting direction is
blocked by a wall or the boundary, the action results in no movement. There is a food source (worm)
located in one of the three right-most locations at the end of each corridor. The agent has an eat
action, which consumes the worm when the agent is at the worm’s location. After the agent consumes
the worm, a new worm appears randomly in one of the other two potential worm locations.
Objective Reward for the Foraging Domain: The designer’s goal is to maximize the average number
of worms eaten per time step. Thus, the objective reward function RO provides a reward of 1.0 when
the agent eats a worm, and a reward of 0 otherwise."

Run `run_experiments.py` to run the experiments.

Visualization of the results is in `visualize_results.ipynb`