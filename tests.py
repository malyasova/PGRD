import numpy as np
#print environment transitions:
def print_env(env):
    def pretty(d, indent=0):
        for key, value in d.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                pretty(value, indent+1)
            else:
                print('\t' * (indent+1) + str(value))
    pretty(env.transitions, indent=1)

def test_value_grad(agent):
    theta = np.random.rand(agent.nS, agent.nA)
    
    delta_theta = 1e-2 * np.random.rand(agent.nS, agent.nA)
    state = 0
    values1, grad1 = agent.plan(state, theta)
    
    values2, grad2 = agent.plan(state, theta + delta_theta)
    
    assert np.allclose(values2 - values1, np.tensordot(grad1, delta_theta, axes=2))
    
def test_policy_grad(agent):
    theta = np.random.rand(agent.nS, agent.nA)
    delta_theta = 1e-3 * np.random.rand(agent.nS, agent.nA)
    state = 0
    
    for action in range(5):
        values1, value_grad1 = agent.plan(state, theta)
        logprobas1 = np.log(agent.policy(values1))
        values2, value_grad2 = agent.plan(state, theta + delta_theta)
        logprobas2 = np.log(agent.policy(values2))
        grad = agent.logpolicy_grad(value_grad1, agent.policy(values1), action)

        assert np.allclose(logprobas2[action] - logprobas1[action], (grad * delta_theta).sum())
    
