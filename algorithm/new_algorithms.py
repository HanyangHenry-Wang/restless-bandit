import numpy as np
import GPy


def Assembly_Line_Scheduling(C, current_pos, line_holder): #current position is 0,1,2...

  num = len(line_holder)
  length =len(line_holder[0])
  e = [-C]*num
  e[current_pos] = 0
  f = [0]*num
  L = np.zeros((num,length-1))

  for i in range(num):
    f[i] = e[i]+line_holder[i][0]  #at time 0
  
  for t in range(1,length):
    f_old = f.copy()
    for i in range(num):
      candidate = [f_old[j]+line_holder[i][t]-C for j in range(num)]
      candidate[i] =  f_old[i]+line_holder[i][t]  
      f[i] = max(candidate) 
      L[i,t-1] = candidate.index(max(candidate) ) 

  f_best = max(f)
  final_pos = f.index(f_best)

  line_res = [final_pos]
  state = int(final_pos)
  

  for t in range(length-1,0,-1):
    choice = int(L[state,t-1])
    line_res.append(choice)
    state = int(L[choice,t-2])


  return line_res[::-1]


def general_DPTS(C,cov_control,*arm_list):
  
  K = len(arm_list)
  T = arm_list[0].T
  sigma = arm_list[0].sigma
  
  choice = []
  arms = []
  reward_paths = []
  
  for i in range(K):
    arm_i = arm_list[i].arm_path
    reward_i=arm_list[i].reward_path()
    
    arms.append(arm_i)
    reward_paths.append(reward_i)
    

  max_reward = np.maximum.reduce(arms)
  Normal=np.sum(np.abs(max_reward))
  regret_holder=np.zeros(T)
  
  GP_models = [] #this is the holder of GP model of arm i
  lengscale_holder = []  #the corresponding lengthscale  
  round_holder= []    #this record the round that arm i is played
  reward_holder= []    #this record the reward that arm i gives


  X_sample=np.array([[0.0]])
  Y_sample=np.array([[0.0]])
  
  for i in range (K):
    arm_i = arm_list[i]
    kernel = GPy.kern.RBF(input_dim=1,variance=arm_i.variance,lengthscale=arm_i.lengthscale)   #using RBF kernel
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(sigma)
    GP_models.append(m)
    
    round_holder.append([])
    reward_holder.append([])

    lengscale_holder.append(arm_i.lengthscale)


  for t in range(T):


    sample_TS=[]
    future_step = int(min(lengscale_holder))
    

    for i in range(K):
      X = np.array(range(t,min(t+future_step,T+1))).reshape(-1,1) 

      mean_temp,cov_temp =  GP_models[i].predict(X,full_cov=True,include_likelihood=False)
      temp = np.random.multivariate_normal(mean_temp.reshape(-1,), cov_control*cov_temp, 1).reshape(-1)    
      sample_TS.append(temp)

    if t==0:   #determine the next pull
      next_pull = np.random.randint(0,K)   
    else:
      path = Assembly_Line_Scheduling(C, choice[-1], sample_TS)
      next_pull = path[0]


    if t >=1 :
      if (next_pull != choice[-1]) : #check whether we switch the arm at time t
        switch = 1
      else:
        switch = 0
    else:
      switch = 0

    regret=max_reward[t]-arms[next_pull][t]    #calculate the regret
    regret_holder[t]=regret+switch*C

    reward= reward_paths[next_pull][t]     #obtain the reward

    round_holder[next_pull].append(t)
    reward_holder[next_pull].append(reward)
    choice.append(next_pull)

    X_sample= np.array(round_holder[next_pull]).reshape(-1,1)     
    Y_sample=np.array(reward_holder[next_pull]).reshape(-1,1)
    

    kernel = GPy.kern.RBF(input_dim=1,variance=arm_list[next_pull].variance,lengthscale=arm_list[next_pull].lengthscale)
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(sigma)

    GP_models[next_pull]=m
    
  Normal_regret = regret_holder/Normal
  

  return Normal_regret,choice





def simulation_DPTS(C,simulation_num,*arm_list):
  
  K = len(arm_list)
  T = arm_list[0].T
  sigma = arm_list[0].sigma
  
  choice = []
  arms = []
  reward_paths = []
  
  for i in range(K):
    arm_i = arm_list[i].arm_path
    reward_i=arm_list[i].reward_path()
    
    arms.append(arm_i)
    reward_paths.append(reward_i)
    

  max_reward = np.maximum.reduce(arms)
  Normal=np.sum(np.abs(max_reward))
  regret_holder=np.zeros(T)
  
  GP_models = [] #this is the holder of GP model of arm i
  lengscale_holder = []  #the corresponding lengthscale  
  round_holder= []    #this record the round that arm i is played
  reward_holder= []    #this record the reward that arm i gives


  X_sample=np.array([[0.0]])
  Y_sample=np.array([[0.0]])
  
  for i in range (K):
    arm_i = arm_list[i]
    kernel = GPy.kern.RBF(input_dim=1,variance=arm_i.variance,lengthscale=arm_i.lengthscale)   #using RBF kernel
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(sigma)
    GP_models.append(m)
    
    round_holder.append([])
    reward_holder.append([])

    lengscale_holder.append(arm_i.lengthscale)


  for t in range(T):


    sample_TS=[]
    future_step = int(min(lengscale_holder))
    
    
    next_choice_holder = np.zeros(simulation_num)  
    
    for ii in range(simulation_num):
      sample_TS=[]
      for i in range(K):
        X = np.array(range(t,min(t+future_step,T+1))).reshape(-1,1) 

        temp = GP_models[i].posterior_samples_f(X,size=1).reshape(-1)
        sample_TS.append(temp)

      if t==0:   #determine the next pull
        next_choice_holder[ii] = np.random.randint(0,K)   
      else:
        path = Assembly_Line_Scheduling(C, choice[-1], sample_TS)
        next_choice_holder[ii] = path[0]
        
    next_choice_holder = next_choice_holder.astype(int)
    counts = np.bincount(next_choice_holder)
    next_pull = np.argmax(counts)
    
    if t >=1 :
      if (next_pull != choice[-1]) : #check whether we switch the arm at time t
        switch = 1
      else:
        switch = 0
    else:
      switch = 0

    regret=max_reward[t]-arms[next_pull][t]    #calculate the regret
    regret_holder[t]=regret+switch*C

    reward= reward_paths[next_pull][t]     #obtain the reward

    round_holder[next_pull].append(t)
    reward_holder[next_pull].append(reward)
    choice.append(next_pull)

    X_sample= np.array(round_holder[next_pull]).reshape(-1,1)     
    Y_sample=np.array(reward_holder[next_pull]).reshape(-1,1)
    

    kernel = GPy.kern.RBF(input_dim=1,variance=arm_list[next_pull].variance,lengthscale=arm_list[next_pull].lengthscale)
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(sigma)

    GP_models[next_pull]=m
    
  Normal_regret = regret_holder/Normal
  

  return Normal_regret,choice