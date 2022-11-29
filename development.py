# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 20:04:45 2022

Iterative Policy Evaluation Algorithm in Python – Reinforcement Learning Tutorial

The tutorial webpage explaining the code is given here:
    
https://aleksandarhaber.com/iterative-policy-evaluation-algorithm-in-python-reinforcement-learning-tutorial/

Author:  Aleksandar Haber
November 2022

"""
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as no
 
env=gym.make("FrozenLake-v1", render_mode="human")
env.reset()
# render the environment
env.render()
env.close()

# observation space - states 
env.observation_space

# actions: left -0, down - 1, right - 2, up- 3
env.action_space


#transition probabilities
#p(s'|s,a) probability of going to state s' 
#          starting from the state s and by applying the action a

# env.P[state][action]
env.P[0][1] #state 0, action 1
# output is a list having the following entries
# (transition probability, next state, reward, Is terminal state?)

# select the discount factor
discountFactor=0.9
# initialize the value function vector
valueFunctionVector=np.zeros(env.observation_space.n)
# maximum number of iterations
maxNumberOfIterations=1000
# convergence tolerance delta
convergenceTolerance=10**(-6)

# convergence list 
convergenceTrack=[]

for iterations in range(maxNumberOfIterations):
    convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
    valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
    for state in env.P:
        outerSum=0
        for action in env.P[state]:
            innerSum=0
            for probability, nextState, reward, isTerminalState in env.P[state][action]:
                #print(probability, nextState, reward, isTerminalState)
                innerSum=innerSum+ probability*(reward+discountFactor*valueFunctionVector[nextState])
            outerSum=outerSum+0.25*innerSum
        valueFunctionVectorNextIteration[state]=outerSum
    if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
        valueFunctionVector=valueFunctionVectorNextIteration
        print('Converged!')
        break
    valueFunctionVector=valueFunctionVectorNextIteration          


# visualize the state values
def grid_print(valueFunction,reshapeDim):
    ax = sns.heatmap(valueFunction.reshape(4,4),
                     annot=True, square=True,
                     cbar=False, cmap='Blues',
                     xticklabels=False, yticklabels=False)
    plt.savefig('valueFunctionGrid.png',dpi=600)
    plt.show()
    
grid_print(valueFunctionVector,4)

plt.plot(convergenceTrack)
plt.xlabel('steps')
plt.ylabel('Norm of the value function vector')
plt.savefig('convergence.png',dpi=600)
plt.show()




