import numpy as np
from agent import Agent
from environment import BirdEnv
from pgrd import PGRD
from gym.utils.seeding import np_random
from multiprocessing import Pool
import os
    

# 5 actions: move right, left, down, up, eat the worm
NUM_ACTIONS = 5 
#the agent observes the full state given by 9*agent_location + worm_location
NUM_STATES = 81
TAU = 100
GAMMA = 0.95
NUM_TRIALS = 32 #not 130 because I only have so much compute
TOTAL_TIMESTEPS = 5000


if __name__ == "__main__":
    rng_env, _ = np_random()
    env = BirdEnv(rng_env)
        
    for depth in range(7):
        print(depth)
        for alpha in [0, 2e-6, 5e-6, 2e-5, 5e-5, 2e-4, 5e-4, 2e-3, 5e-3, 1e-2]:
            print(alpha)
            for beta in [0, 0.4, 0.7, 0.9, 0.95, 0.99]:
                def run_trial(num_trial):
                    env.reset()
                    rng_agent, _ = np_random()
                    agent = Agent(depth, TAU, GAMMA, rng_agent, NUM_ACTIONS, NUM_STATES)
                    model = PGRD(agent, env, alpha, beta)
                    return model.learn(total_timesteps=TOTAL_TIMESTEPS, visualize=False)
                
                pool = Pool(os.cpu_count())
                try:
                    returns = pool.map(run_trial, np.arange(num_trials))
                    returns = np.sum(np.array(returns), axis=0) / num_trials
                finally:
                    pool.close()
                    pool.join()
                np.save("results/Result_depth_{}_alpha_{}_beta_{}.npy".format(depth, alpha, beta), returns)
