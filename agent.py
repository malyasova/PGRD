import numpy as np
from collections import defaultdict
from functools import lru_cache

def softmax(action_values, tau):
    """
    Arguments: action_values - 1-dimensional array
    tau - temperature
    """
    preferences = action_values * tau
    max_preference = np.max(preferences)
    exp_prefs = np.exp(preferences - max_preference)
    return exp_prefs / np.sum(exp_prefs)

class Agent:
    def __init__(self, depth, tau, gamma, rng, nA, nS):
        self.nA = nA
        self.nS = nS
        self.depth = depth #depth of planning
        self.tau = tau #policy temperature
        self.gamma = gamma #discount rate
        #agent's model of the environment
        #N[s][a] =  {total: total_visits, 'counts': {s': x, ...}}
        #N[s][a][s'] - number of visits to s' after taking action s in state a
        #N[s][a][s`] / N[s][a][total] = Pr(s`|s, a)
        self.N = defaultdict(lambda: defaultdict(lambda: {'total':0, 'counts': defaultdict(lambda:0)}))
        self.rand_generator = rng
        
    def update(self, state, action, newstate):
        self.N[state][action]['total'] += 1
        self.N[state][action]['counts'][newstate] += 1

    def plan(self, state, theta):
        """ Compute d-step Q-value function and its theta-gradient at state""" 
        @lru_cache(maxsize=None)
        def _plan(self, state, d):
            """ Recursive memoized function"""
            reward_grad = np.zeros((self.nA, self.nS, self.nA))
            for a in range(self.nA):
                reward_grad[a,state,a] = 1
            if d == 0:
                action_values = theta[state]
                value_grad = reward_grad
            else:
                inc = np.zeros(self.nA)
                grad_inc = np.zeros((self.nA, self.nS, self.nA))
                for action in self.N[state].keys():
                    for state_next, count in self.N[state][action]['counts'].items():
                        values_next, grad_next = _plan(self, state_next, d-1)
                        action_next = np.argmax(values_next)
                        p = count / self.N[state][action]['total']
                        inc[action] += values_next[action_next] * p
                        grad_inc[action, state_next, action_next] += np.argmax(values_next) * p

                action_values = theta[state] + self.gamma * inc
                value_grad = reward_grad + self.gamma * grad_inc
            return action_values, value_grad
        return _plan(self, state, self.depth)
    
    def logpolicy_grad(self, value_grad, probas, action):
        """
        Arguments: 
        value_grad: nA x nS x nA
        probas: nA
        action: int
        Returns:
        grad: nS x nA
        """
        grad = self.tau * (value_grad[action] - np.tensordot(probas, value_grad, axes=1))
        return grad
    
    def policy(self, action_values):
        probas = softmax(action_values, self.tau)
        return probas

    def step(self, state, theta):
        action_values, value_grad = self.plan(state, theta)
        # compute the Boltzman stochastic policy parametrized by action_values
        probas = self.policy(action_values) #shape: nA
        # select action according to policy
        action = self.rand_generator.choice(np.arange(self.nA), p=probas)
        grad = self.logpolicy_grad(value_grad, probas, action)
        return action, grad
    
class Agent2:
    """
    Tried implementing environment model as a matrix instead of dictionary,
    result: this implementation runs slower.
    """
    def __init__(self, depth, tau, gamma, rng, nA, nS):
        self.nA = nA
        self.nS = nS
        self.depth = depth #depth of planning
        self.tau = tau #policy temperature
        self.gamma = gamma #discount rate
        #agent's model of the environment
        #N[s, a, s`] = number of visits to s` after taking action a in state s
        #N[s, a, s`] / N[s, a, :].sum() = T; T[s, a, s'] = Pr(s`|s, a)
        self.N = np.ones((nS, nA, nS)) * 1e-10
        self.rand_generator = rng
        
    def update(self, state, action, newstate):
        self.N[state, action, newstate] += 1
        
    def plan(self, state, theta):
        """ Compute d-step Q-value function and its theta-gradient"""
        #transition probabilities:
        T = self.N / self.N.sum(axis=2, keepdims=True) 
        # Q_0 = theta
        action_values = theta # nS x nA matrix
        #value_grad[s, a, s', a'] = d Q(s,a) / d theta(s', a')
        reward_grad = np.zeros((self.nS, self.nA, self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                reward_grad[s,a,s,a] = 1
                
        value_grad = reward_grad
        for _ in range(self.depth):
            #compute action value gradient:
            # T: nS x nA x nS, value_grad[.., ..] : nS x nS x nA
            grad_inc = np.tensordot(T, value_grad[np.arange(self.nS), action_values.argmax(axis=1)], axes=1)
            value_grad = reward_grad + self.gamma * grad_inc

            #update action values
            action_values = theta + self.gamma * T.dot(action_values.max(axis=1))

            assert action_values.shape == (self.nS, self.nA)
        return action_values[state], value_grad[state]
            
    def logpolicy_grad(self, value_grad, probas, action):
        """
        Arguments: 
        value_grad: nA x nS x nA
        probas: nA
        action: int
        Returns:
        grad: nS x nA
        """
        grad = self.tau * (value_grad[action] - np.tensordot(probas, value_grad, axes=1))
        return grad
    
    def step(self, state, theta):
        action_values, value_grad = self.plan(state, theta)
        # compute the Boltzman stochastic policy parametrized by action_values
        probas = self.policy(action_values) #shape: nA
        # select action according to policy
        action = self.rand_generator.choice(np.arange(self.nA), p=probas)
        grad = self.logpolicy_grad(value_grad, probas, action)
        return action, grad
        
    def policy(self, action_values):
        probas = softmax(action_values, self.tau)
        return probas
