from algorithm.algorithms import * 
from bandit_process.arm_generator import *
from run_experiment.exp_discount import experiment2
import sys


def temp():
    
    discount_factor_holder=[1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.]
    
    for df in discount_factor_holder:
    
    
        
        regret_TS = []
        regret_mean = []
       

        C = 0.25
        N = 250

    

        for exp in range(N):

            T=200

            arm1=expect_reward_generator(T,lengthscale=8,variance=5,Smooth=True,Plot=False)
            arm2=expect_reward_generator(T,lengthscale=10,variance=5,Smooth=True,Plot=False)
            
            Normal = np.sum(np.abs(np.maximum.reduce([arm1,arm2])))

            regret_holder1,_,_=GPR_DP(T,C, 0, arm1,arm2, discount_factor=df,TS=True)
            regret_holder2,_,_=GPR_DP(T,C, 0, arm1,arm2, discount_factor=df,TS=False)
            
            regret_TS.append(regret_holder1/Normal)
            regret_mean.append(regret_holder2/Normal)
            
        regret_record1 = np.array(regret_TS).reshape(N,T)
        regret_record2 = np.array(regret_mean).reshape(N,T)


        np.savetxt('DP_cost0.25_discount_'+str(df)+'.csv', regret_record1, delimiter=',')
        np.savetxt('mean_DP_cost0.25_discount_'+str(df)+'.csv', regret_record2, delimiter=',')

       
        



if __name__=='__main__':
    
   res = temp()
  
    

