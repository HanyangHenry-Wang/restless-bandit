from algorithm.algorithms import * 
from bandit_process.arm_generator import *
import os

os.mkdir('TEMP') 

control_holder = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0 ]

for ctrl in control_holder:
    
    T = 100
    N = 1
    C = 0.75

        
    regret_holder = []
    choice_holder = []



    for exp in range(N):

        arm1=expect_reward_generator(200,lengthscale=8,variance=5,Smooth=True,Plot=False)
        arm2=expect_reward_generator(200,lengthscale=10,variance=5,Smooth=True,Plot=False)
        
        Normal = np.sum(np.abs(np.maximum.reduce([arm1,arm2])))

        regret,choice = GPR_generalTS2(T,C, arm1,arm2,control=ctrl)
        
        regret_holder.append(regret/Normal)
        choice_holder.append(choice)
        
    regret_record = np.array(regret_holder).reshape(N,T)  
    choice_record = np.array(choice_holder).reshape(N,T)  

    np.savetxt('TEMP/GeneralTS_SC0.75_control'+str(ctrl)+'.csv', regret_record, delimiter=',')
    np.savetxt('TEMP/choice_GeneralTS_SC0.75_control'+str(ctrl)+'.csv', choice_record, delimiter=',')