from RestlessBandit.algorithms import *
from RestlessBandit.arm_generator import *

def run():
    
    discount_factor_holder=[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    for df in discount_factor_holder:
  
        regret_normal = []
        regret_dp = []

        switch_normal = []
        switch_dp = []

        C = 5
        N = 2

        

        for exp in range(N):

            T=200

            arm1=expect_reward_generator(T,lengthscale=8,variance=5,Smooth=True,Plot=False)
            arm2=expect_reward_generator(T,lengthscale=10,variance=5,Smooth=True,Plot=False)
            
            Normal = np.sum(np.abs(np.maximum.reduce([arm1,arm2])))


            regret_holder2,_,_=GPR_DP(T,C, 0, arm1,arm2, discount_factor=df)
            regret_dp.append(regret_holder2/Normal)
            
            
        regret_record2=np.array(regret_dp).reshape(N,T)


        np.savetxt('DP_cost5_discount_'+str(df)+'.csv', regret_record2, delimiter=',')
       
        



if __name__=='__main__':
    run()

