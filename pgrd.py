import numpy as np
class PGRD:
    def __init__(self, agent, env, alpha, beta):
        self.agent = agent
        self.env = env
        self.alpha = alpha #step size
        self.beta = beta 
        #theta is initialized so that the initial reward function = objective reward function
        self.theta = np.zeros((env.nS, env.nA))
        for cell in [2,5,8]:
            self.theta[10*cell, 4] = 1
        #variable to store theta gradient
        self.z = np.zeros((env.nS, env.nA))

    def learn(self, total_timesteps, visualize=False):
        state = self.env.state()
        total_reward = 0
        returns = []
        for i in range(total_timesteps):
            if visualize:
                print(i)
                self.env.render()
            action, grad = self.agent.step(state, self.theta)
            newstate, reward = self.env.step(action)
            total_reward += reward
            #update agent's model of the environment:
            self.agent.update(state, action, newstate)
            state = newstate
            #update theta
            self.z = self.beta * self.z + grad
            self.theta += self.alpha * reward * self.z
            #cap parameters at +-1:
            self.theta = np.maximum(self.theta, -1)
            self.theta = np.minimum(self.theta, 1)
            returns.append(total_reward / (i+1))
        return np.array(returns)
