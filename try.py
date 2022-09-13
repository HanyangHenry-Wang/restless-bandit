import os
from algorithm.new_algorithms import * 
from bandit_process.new_arms import *
from matplotlib import pyplot as plt


C_holder = [0,1.5,30,40,60,80]


for C in C_holder:
  
  regret_DPTS = []
  choice_DPTS = []


  N = 250
  T = 400
  
  

  for exp in range(N):


    arm1=GP_Arm(lengthscale=6, variance=15,sigma=0.1, T=400)
    arm2=GP_Arm(lengthscale=6, variance=15,sigma=0.1, T=400)
    arm3=GP_Arm(lengthscale=18, variance=4,sigma=0.1, T=400)
    arm4=GP_Arm(lengthscale=18, variance=4,sigma=0.1, T=400)
 
    regret_holder1,choice_holder1 = simulation_DPTS(C,25, arm1,arm2,arm3,arm4)
    
    

    regret_DPTS.append(regret_holder1)
    choice_DPTS.append(choice_holder1)
  
    
    

  regret_record1 = np.array(regret_DPTS).reshape(N,T)
  choice_DPTS = np.array(choice_DPTS)

  np.savetxt('simulation_4arms_regret_DPTS_cost'+str(C)+'.csv', regret_record1, delimiter=',')
  np.savetxt('simulation_4arms_choice_DPTS_cost'+str(C)+'.csv', choice_DPTS, delimiter=',')