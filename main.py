from RestlessBandit.algorithms import *
from RestlessBandit.arm_generator import *

def run():
    
    T=200
    K=3


    arm1 = expect_reward_generator(T,lengthscale=10,variance=5,Smooth=True,Plot=True)
    arm2 = expect_reward_generator(T,lengthscale=20,variance=5,Smooth=True,Plot=True)
    arm3 = expect_reward_generator(T,lengthscale=20,variance=5,Smooth=True,Plot=True)
    
    
    C=np.sum(np.abs(np.maximum.reduce([arm1,arm2,arm3])))
    
    regret = EF(T, arm1,arm2,arm3)/C
    
    return regret
        



if __name__=='__main__':
    run()

