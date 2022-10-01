import os
from algorithm.new_algorithms import * 
from bandit_process.new_arms import *


C_holder = [0,0.25,0.75,1.5,5]

for C in C_holder:
  
  regret_DPS_s5 = []
  choice_DPS_s5 = []
  regret_DPS_s10 = []
  choice_DPS_s10 = []
  regret_DPS_s25 = []
  choice_DPS_s25 = []
  regret_DPTS = []
  choice_DPTS = []
  regret_DPPM = []
  choice_DPPM = []


  N = 250
  T = 200
  


  for exp in range(N):
    
    arm1=GP_Arm(lengthscale=8, variance=5,sigma=0.1, T=200)
    arm2=GP_Arm(lengthscale=10, variance=5,sigma=0.1, T=200)
 
    regret_holder1,choice_holder1 = simulation_DPTS(C,5, arm1,arm2)
    regret_holder2,choice_holder2 = simulation_DPTS(C,10, arm1,arm2)
    regret_holder3,choice_holder3 = simulation_DPTS(C,25, arm1,arm2)
    regret_holder4,choice_holder4= general_DPTS(C,1,arm1,arm2) #DPTS
    regret_holder5,choice_holder5 = general_DPTS(C,0,arm1,arm2) #DPPM
    
    

    regret_DPS_s5.append(regret_holder1)
    choice_DPS_s5.append(choice_holder1)
    regret_DPS_s10.append(regret_holder2)
    choice_DPS_s10.append(choice_holder2)
    regret_DPS_s25.append(regret_holder3)
    choice_DPS_s25.append(choice_holder3)
    regret_DPTS.append(regret_holder4)
    choice_DPTS.append(choice_holder4)
    regret_DPPM.append(regret_holder5)
    choice_DPPM.append(choice_holder5)
  
    

  regret_DPS_s5 = np.array(regret_DPS_s5).reshape(N,T)
  choice_DPS_s5 = np.array(choice_DPS_s5)
  regret_DPS_s10 = np.array(regret_DPS_s10).reshape(N,T)
  choice_DPS_s10 = np.array(choice_DPS_s10)
  regret_DPS_s25 = np.array(regret_DPS_s25).reshape(N,T)
  choice_DPS_s25 = np.array(choice_DPS_s25)
  regret_DPTS = np.array(regret_DPTS).reshape(N,T)
  choice_DPTS = np.array(choice_DPTS)
  regret_DPPM = np.array(regret_DPPM).reshape(N,T)
  choice_DPPM = np.array(choice_DPPM)
 

  np.savetxt('2 arm experiments/simulation_2arms_regret_DPS_s5_cost'+str(C)+'.csv', regret_DPS_s5, delimiter=',')
  np.savetxt('2 arm experiments/simulation_2arms_choice_DPS_s5_cost'+str(C)+'.csv', choice_DPS_s5, delimiter=',')
  np.savetxt('2 arm experiments/simulation_2arms_regret_DPS_s10_cost'+str(C)+'.csv', regret_DPS_s10, delimiter=',')
  np.savetxt('2 arm experiments/simulation_2arms_choice_DPS_s10_cost'+str(C)+'.csv', choice_DPS_s10, delimiter=',')
  np.savetxt('2 arm experiments/simulation_2arms_regret_DPS_s25_cost'+str(C)+'.csv', regret_DPS_s25, delimiter=',')
  np.savetxt('2 arm experiments/simulation_2arms_choice_DPS_s25_cost'+str(C)+'.csv', choice_DPS_s25, delimiter=',')
  np.savetxt('2 arm experiments/2arms_regret_DPTS_cost'+str(C)+'.csv', regret_DPTS, delimiter=',')
  np.savetxt('2 arm experiments/2arms_choice_DPTS_cost'+str(C)+'.csv', choice_DPTS, delimiter=',')
  np.savetxt('2 arm experiments/2arms_regret_DPPM_cost'+str(C)+'.csv', regret_DPPM, delimiter=',')
  np.savetxt('2 arm experiments/2arms_choice_DPPM_cost'+str(C)+'.csv', choice_DPPM, delimiter=',')