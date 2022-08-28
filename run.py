from algorithm.algorithms import * 
from bandit_process.arm_generator import *
from run_experiment.exp_discount import experiment2
import sys


def temp():
    
    C_holder = [0, 0.25, 0.75, 1.5, 5, 7.5, 10, 20, 30]
    
    for C in C_holder:
  
        choice_holder1 = []
        choice_holder2 = []
        choice_holder3 = []
        
        N = 250



        for exp in range(N):

            T=200

            arm1=expect_reward_generator(T,lengthscale=8,variance=5,Smooth=True,Plot=False)
            arm2=expect_reward_generator(T,lengthscale=16,variance=5,Smooth=True,Plot=False)
            
            Normal = np.sum(np.abs(np.maximum.reduce([arm1,arm2])))

            regret_holder1,choice1=GPR_DP(T,C, 0, arm1,arm2,TS=True)
            regret_holder2,choice2=GPR_DP(T,C, 0, arm1,arm2,TS=False)
            regret_holder3,choice3 = GPR_fit(T, 'RBF', C, arm1, arm2)
            
            choice_holder1.append(choice1)
            choice_holder2.append(choice2)
            choice_holder3.append(choice3)
        
        
            
        choice_record1 = np.array(choice_holder1).reshape(N,T)  
        choice_record2 = np.array(choice_holder2).reshape(N,T)  
        choice_record3 = np.array(choice_holder3).reshape(N,T)  
            
        np.savetxt('DPTS_cost'+str(C)+'.csv', choice_record1, delimiter=',')
        np.savetxt('DPPM_cost'+str(C)+'.csv', choice_record2, delimiter=',')
        np.savetxt('GPRTS_cost'+str(C)+'.csv', choice_record3, delimiter=',')
        

       
        



if __name__=='__main__':
    
   res = temp()
  
    

