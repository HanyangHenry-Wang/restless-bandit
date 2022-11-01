import numpy as np
from matplotlib import pyplot as plt
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
import GPy
import math

class Arm:
    def __init__(self, sigma=0.1, T=200):
        
        self.sigma = sigma
        self.T = T
    
    
    def reward_path(self):  #return the reward of one arm
        noise = (np.random.normal(0,self.sigma,self.T)).reshape(-1,1)
        reward = self.arm_path+noise
        
        return reward


class GP_Arm(Arm):
    
     def __init__(self, lengthscale=10, variance=5,sigma=0.1, T=200):
        super(GP_Arm,self).__init__(sigma=sigma, T=T) 
        self.lengthscale = lengthscale
        self.variance = variance
        self.arm_path = self.generate_arm()
        
     def generate_arm(self):
        
        X_sample=np.array([[0.0]])
        Y_sample=np.array([[0.0]])

        kernel = GPy.kern.RBF(input_dim=1,variance=self.variance,lengthscale=self.lengthscale) #here we use RBF kernel to generate arm
        m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
        m.Gaussian_noise.variance.fix(0.0)

        X = np.array(range(self.T)).reshape(-1,1)   
        expected_reward = m.posterior_samples_f(X,size=1).reshape(-1,1)
        
        return expected_reward