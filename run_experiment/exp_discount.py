from restless_bandit.algorithms import * 
from restless_bandit.arm_generator import *


def experiment2(C):
    
    df = [0.5,0.1]
    
    for df in df:
  
        
        regret_dp = []

        N = 2

        for exp in range(N):

            T=200

            arm1=expect_reward_generator(T,lengthscale=8,variance=5,Smooth=True,Plot=False)
            arm2=expect_reward_generator(T,lengthscale=10,variance=5,Smooth=True,Plot=False)
            
            Normal = np.sum(np.abs(np.maximum.reduce([arm1,arm2])))


            regret_holder2,_,_=GPR_DP(T,C, 0, arm1,arm2, discount_factor=df)
            regret_dp.append(regret_holder2/Normal)
            
            print('haha')
            
            
        regret_record2=np.array(regret_dp).reshape(N,T)
        regret_cumsum2=np.cumsum(regret_record2,axis=1)
        
        
       


        #np.savetxt(f'DP_cost{C}_discount_str{df}.csv', regret_record2, delimiter=',')