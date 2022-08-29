from algorithm.algorithms import * 
from bandit_process.arm_generator import *


def exp3():
    
    C_holder = [0, 0.75, 1.5, 5 ]

    for C in C_holder:
    
        regret_GPRTS = []
        regret_DPTS = []
        regret_DPPM = []

        N = 1 #250


        for exp in range(N):

            T=200

            arm1=expect_reward_generator(T,lengthscale=8,variance=5,Smooth=True,Plot=False)
            arm2=expect_reward_generator(T,lengthscale=10,variance=5,Smooth=True,Plot=False)
            
            Normal = np.sum(np.abs(np.maximum.reduce([arm1,arm2])))

            regret_holder1,_=GPR_fit(T, 'RBF',C,arm1,arm2)
            regret_holder2,_=GPR_DP(T,C, arm1,arm2, TS= True)
            regret_holder3,_=GPR_DP(T,C, arm1,arm2, TS= False)
            

            regret_GPRTS.append(regret_holder1/Normal)
            regret_DPTS.append(regret_holder2/Normal)
            regret_DPPM.append(regret_holder3/Normal)
            

        regret_record1 = np.array(regret_GPRTS).reshape(N,T)
        regret_record2 = np.array(regret_DPTS).reshape(N,T)
        regret_record3 = np.array(regret_DPPM).reshape(N,T)

        # np.savetxt('result/experiment3 data/GPRTS_cost'+str(C)+'.csv', regret_record1, delimiter=',')
        # np.savetxt('result/experiment3 data/DPTS_cost'+str(C)+'.csv', regret_record2, delimiter=',')
        # np.savetxt('result/experiment3 data/DPPM_cost'+str(C)+'.csv', regret_record3, delimiter=',')
        
    print('experiment 3 finished!')
