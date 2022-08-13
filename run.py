import constant as C
from method_Gaussian_reward import *
from method_Bernoulli_rewards import *
from progressbar import ProgressBar
import os 

def run():
    T=200
    K=3


    arm1=expect_reward_generator(T,lengthscale=10,variance=5,Smooth=True,Plot=True)
    arm2=expect_reward_generator(T,lengthscale=20,variance=5,Smooth=True,Plot=True)
    
    a,b,c,d,e = EF_plotting(200, arm1,arm2)
    print('okay')
        



if __name__=='__main__':
    run()

