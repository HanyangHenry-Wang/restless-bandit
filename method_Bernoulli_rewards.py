import numpy as np
from matplotlib import pyplot as plt
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
import GPy
import math
import random
from scipy.stats import beta


def Sigmoid(x):
  return math.exp(x)/(math.exp(x)+1)

  
def expect_reward_generator_Bernoulli(T,lengthscale=20,variance=5,Plot=True): #this function generates the expected reward 


    X_sample=np.array([[0.0]])
    Y_sample=np.array([[0.0]])

    kernel = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(0.0)

    X = np.array(range(T)).reshape(-1,1)   
    expected_reward = m.posterior_samples_f(X,size=1).reshape(-1,1)

    er = np.zeros(T)  #np.sin(expected_reward)

    for i,v in enumerate(expected_reward):
      er[i] = Sigmoid(v)

   

    if Plot: 
      plt.plot(er)
      #plt.plot(expected_reward)

    return er



def Bernoulli_reward(theta):

  if theta<0 or theta>1:
    print('theta is out of range')

  u= random.random()

  if u<theta:
    reward = 1
  else:
    reward = 0

  return reward


def reward_generator_Bernoulli (round,expected_reward): #this function generates the reward 
  '''
  round is the time step
  expected_reward is the bandit process
  return the stochastic reward
  '''
  theta= expected_reward[round] 

  if theta<0 or theta>1:
    print('theta is out of range')

  reward = Bernoulli_reward(theta)
  

  return reward



def Dynamic_Thompson_Sampling(T, *arms):

  max_reward = np.maximum.reduce(arms)
  regret_holder=np.zeros(T)
  K = len(arms)
  C= int(T/10)

  S_and_F = []
  for i in range(K):
    S_and_F.append([0.5,0.5])

  for t in range(T):

    sample_holder=np.zeros(K)

    for i in range(K):
      sample_holder[i] = np.random.beta(S_and_F[i][0], S_and_F[i][1])

    next_pull=np.argmax(sample_holder)    #determine the next arm to pull
    regret=max_reward[t]-arms[next_pull][t]  #calculate the regret
    regret_holder[t]=regret

    theta = arms[next_pull][t]
    reward = Bernoulli_reward(theta)

    if (S_and_F[next_pull][0]+S_and_F[next_pull][1])<C:
      S_and_F[next_pull][0] = S_and_F[next_pull][0] +reward
      S_and_F[next_pull][1] = S_and_F[next_pull][1] +1-reward
    
    else:
      S_and_F[next_pull][0] = (S_and_F[next_pull][0] +reward)*(C/(C+1))
      S_and_F[next_pull][1] = (S_and_F[next_pull][1] +1-reward)*(C/(C+1))

  return regret_holder





def Discounted_Thompson_Sampling(T, *arms):
  
  K = len(arms)
  max_reward = np.maximum.reduce(arms)
  regret_holder=np.zeros(T)

  gamma= 0.8

  alpha0=0.5
  beta0=0.5

  S_and_F = []
  for i in range(K):
    S_and_F.append([0.,0.])

  for t in range(T):

    sample_holder=np.zeros(K)
    
    for i in range(K):
      sample_holder[i] = np.random.beta(S_and_F[i][0]+alpha0, S_and_F[i][1]+beta0)

    next_pull=np.argmax(sample_holder) #determine the next arm to pull
    regret=max_reward[t]-arms[next_pull][t]#calculate the regret
    regret_holder[t]=regret

    theta = arms[next_pull][t]
    reward = Bernoulli_reward(theta)

    for i in range(K):

      if i!=next_pull:
        S_and_F[i][0]= gamma*S_and_F[i][0]
        S_and_F[i][1]= gamma*S_and_F[i][1]
      elif i==next_pull:
        S_and_F[i][0]= reward+gamma*S_and_F[i][0]
        S_and_F[i][1]= (1-reward)+gamma*S_and_F[i][1]


  return regret_holder



def DTS_with_GP(T, *arms):

  K = len(arms)
  max_reward = np.maximum.reduce(arms)
  regret_holder=np.zeros(T)
  
  round_holder=[]
  mean_holder=[]
  GP_value_holder= []
  S_and_F = []
  GP_models = []


  X_sample=np.array([[0.0]])
  Y_sample=np.array([[np.arcsin(0.5)]])

  for i in range(K):
      round_holder.append([])
      mean_holder.append([])
      GP_value_holder.append([])
      S_and_F.append([0.,0.])
      
      kernel1 = GPy.kern.Matern52(input_dim=1)
      m1 = GPy.models.GPRegression(X_sample,Y_sample,kernel1)
      m1.Gaussian_noise.variance.fix(0.1)
      GP_models.append(m1)

  gamma= 0.8
  alpha0=0.5
  beta0=0.5



  for t in range(T):

    sample_holder=np.zeros(K)
    X =  np.array( (t)).reshape(-1,1) 

    for i in range(K):
      sample_holder[i] = math.sin(       float(GP_models[i].posterior_samples_f(X,size=1).reshape(-1,1)) )  #since sigmoid is increasing, we picking the highest value from GP is equal to picking the highest theta
    
    next_pull=np.argmax(sample_holder)    #determine the next arm to pull
    regret=max_reward[t]-arms[next_pull][t]   #calculate the regret
    regret_holder[t]=regret

    theta = arms[next_pull][t]
    reward = Bernoulli_reward(theta)


    for i in range(K):  #renew

      if i!=next_pull:
        S_and_F[i][0]= gamma*S_and_F[i][0]
        S_and_F[i][1]= gamma*S_and_F[i][1]
      elif i==next_pull:
        S_and_F[i][0]= reward+gamma*S_and_F[i][0]
        S_and_F[i][1]= (1-reward)+gamma*S_and_F[i][1]

    round_holder[next_pull].append(t)

    mean_est = (S_and_F[next_pull][0]+alpha0)/(S_and_F[next_pull][0]+alpha0+S_and_F[next_pull][1]+beta0)
    mean_holder[next_pull].append(mean_est)
    GP_value_holder[next_pull].append(np.arcsin(mean_est))

    X_sample= np.array(round_holder[next_pull]).reshape(-1,1)      
    Y_sample=np.array(GP_value_holder[next_pull]).reshape(-1,1)
    
    
    kernel = GPy.kern.Matern52(input_dim=1)
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.optimize()

    GP_models[next_pull]=m




  return regret_holder