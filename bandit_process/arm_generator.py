import numpy as np
from matplotlib import pyplot as plt
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
import GPy
import math


def expect_reward_generator(T,lengthscale=20,variance=2,Smooth=True,Plot=True): #this function generates the expected reward 
  
  """this function generates the expected reward in a Gaussian process form

  Args:
      T (int): total time steps
      lengthscale (float,optional): _description_. Defaults to 20.
      variance (float, optional): _description_. Defaults to 2.
      Smooth (bool, optional): we can choose to generate a Gaussian process or a Brownian motion. Defaults to True.
      Plot (bool, optional): whether we would like to plot the bandit process. Defaults to True.

  Returns:
      array:  return the bandit prceoss
  """  

 
  if Smooth:

    X_sample=np.array([[0.0]])
    Y_sample=np.array([[0.0]])

    kernel = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(0.0)

    X = np.array(range(T)).reshape(-1,1)   
    expected_reward = m.posterior_samples_f(X,size=1).reshape(-1,1)
    

  else:
    
    increments=np.random.randn(T)* math.sqrt(variance)  
    expected_reward=np.cumsum(increments).reshape(-1,1)
  
  if Plot: 
      plt.plot(expected_reward)
  
  
  return expected_reward


def expect_reward_generato_sin(T,period,hight,shift,Plot=True): 
  
  """
  this function generates the expected reward in a sin function form

  Args:
      T (int): _description_
      period (float): one parameter in the sin function
      hight (float): one parameter in the sin function
      shift (float): one parameter in the sin function
      Plot (bool, optional): whether to plot the arm process. Defaults to True.

  Returns:
      array:  return the bandit prceoss
  """  
  
  time = np.array(range(T))
  expected_reward = hight*np.sin((time+shift)/period)

  if Plot is True:
    plt.plot(time,expected_reward)
    plt.show()

  
  
  return expected_reward



def reward_generator(round,expected_reward,sigma=0.1):
  
  """this function generates the reward 

  Args:
      round (_type_): _description_
      expected_reward (_type_): _description_
      sigma (float, optional): _description_. Defaults to 0.1.

  Returns:
      float: a reward
  """  
  
  mu= expected_reward[round] 
  reward = float(np.random.normal(mu,sigma,1))
  

  return reward