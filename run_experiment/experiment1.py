from algorithm.algorithms import * 
from bandit_process.arm_generator import *

def exp1():
    
    N = 1 #100
    T = 200

    DTS_record=[]
    UCB_record=[]
    GPR_record=[]



    for exp in range(N):

        arm1=expect_reward_generator(T,lengthscale=8,variance=5,Smooth=True,Plot=False)
        arm2=expect_reward_generator(T,lengthscale=16,variance=5,Smooth=True,Plot=False)
        arm3=expect_reward_generator(T,lengthscale=32,variance=5,Smooth=True,Plot=False)

       
        C=np.sum(np.abs(np.maximum.reduce([arm1,arm2,arm3])))

        regret_holder_DTS=DTS(T, arm1,arm2,arm3)/C
        regret_holder_UCB= UCB_f(T, arm1,arm2,arm3)/C
        regret_holder_GPR_mismatch,_= GPR_fit(T, 'Matern52',0, arm1,arm2,arm3)/C

        DTS_record.append(regret_holder_DTS)
        UCB_record.append(regret_holder_UCB)
        GPR_record.append(regret_holder_GPR_mismatch)
    
    DTS_record=np.array(DTS_record).reshape(N,T)
    UCB_record=np.array(UCB_record).reshape(N,T)
    GPR_record=np.array(GPR_record).reshape(N,T)


    # np.savetxt('result/experiment1 data/DTS_record.csv', DTS_record, delimiter=',')
    # np.savetxt('result/experiment1 data/UCB_record.csv', UCB_record, delimiter=',')
    # np.savetxt('result/experiment1 data/GPR_record.csv', GPR_record, delimiter=',')
    
    print('experiment 1 finished!')