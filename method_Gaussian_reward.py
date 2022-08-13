import numpy as np
from matplotlib import pyplot as plt
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
import GPy
import math
import constant as C



def expect_reward_generator(T,lengthscale=20,variance=2,Smooth=True,Plot=True): #this function generates the expected reward 
  
  if Smooth:

    X_sample=np.array([[0.0]])
    Y_sample=np.array([[0.0]])

    kernel = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(0.0)

    X = np.array(range(T)).reshape(-1,1)   
    expected_reward = m.posterior_samples_f(X,size=1).reshape(-1,1)
    

  else:
    
    increments=np.random.randn(T)* math.sqrt(variance)  
    expected_reward=np.cumsum(increments).reshape(-1,1)
  
  if Plot: 
      plt.plot(expected_reward)
  
  
  return expected_reward


def expect_reward_generato_sin(T,period,hight,shift,Plot=True): #this function generates the expected reward 
  
  time = np.array(range(T))
  expected_reward = hight*np.sin((time+shift)/period)

  if Plot is True:
    plt.plot(time,expected_reward)
    plt.show()

  
  
  return expected_reward



def reward_generator(round,expected_reward,sigma=0.1): #this function generates the reward 
  '''
  round is the time step
  expected_reward is the bandit process
  return the stochastic reward
  '''
  mu= expected_reward[round] 

  reward = float(np.random.normal(mu,sigma,1))
  

  return reward




def BM_calibration(round_holder,reward_holder,sigma=0.1):

  '''
  this function calculates sigma_i^2
  '''

  kernel = GPy.kern.Brownian(input_dim=1)
  m = GPy.models.GPRegression(round_holder.reshape(-1,1),reward_holder.reshape(-1,1),kernel)
  m.Gaussian_noise.variance.fix(sigma)
  m.optimize()

  var=m.Brownian.variance[0]

  return var


def EF(T, *arms):

  K = len(arms)
  max_reward = np.maximum.reduce(arms)
  choice = []

  gamma= 0.5

  mu_holder=[]
  sigma_holder=[]
  R_holder=[] #the time that arm i has not been pulled
  round_holder=[]
  reward_holder=[]

  for i in range(K):
    mu_holder.append(0.)
    sigma_holder.append(1.)
    R_holder.append(1) #the time that arm i has not been pulled
    round_holder.append([])
    reward_holder.append([])

  regret_holder=np.zeros(T)

  for t in range(T):

    sample_TS=np.zeros(K)
    
    for i in range(K):
      sample_TS[i]=float(np.random.normal(mu_holder[i],sigma_holder[i]*math.sqrt(R_holder[i]),1))

    next_pull=np.argmax(sample_TS) #determine the next arm to pull
    regret=max_reward[t]-arms[next_pull][t]#calculate the regret
    regret_holder[t]=regret
    

    reward=reward_generator(t,arms[next_pull],sigma=0.1) #obtain the reward
    mu_new=gamma**R_holder[next_pull]*mu_holder[next_pull]+(1-gamma**R_holder[next_pull])*reward
    mu_holder[next_pull]=mu_new

    round_holder[next_pull].append(t)
    reward_holder[next_pull].append(reward)
    choice.append(next_pull)

    for i in range(K):
      if i== next_pull:
        R_holder[i]=1
      else:
        R_holder[i]=R_holder[i]+1

    #print(R_holder[0])

    sigma_holder[next_pull]=math.sqrt(BM_calibration(np.array(round_holder[next_pull]),np.array(reward_holder[next_pull]),sigma=0.1))

  return regret_holder




def GPR_fit(T,model,C,*arms):

  choice = []


  max_reward = np.maximum.reduce(arms)
  K = len(arms)

  X_sample=np.array([[0.0]])
  Y_sample=np.array([[0.0]])

  GP_models = []

  if model == 'RBF':
  
    lengthscale=9
    variance=3

    for i in range (K):
      kernel1 = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)
      m1 = GPy.models.GPRegression(X_sample,Y_sample,kernel1)
      m1.Gaussian_noise.variance.fix(0.1)
      GP_models.append(m1)

  elif model == 'Matern52':

    lengthscale=9
    variance=3

    for i in range (K):
      kernel1 = GPy.kern.Matern52(input_dim=1,variance=variance,lengthscale=lengthscale)
      m1 = GPy.models.GPRegression(X_sample,Y_sample,kernel1)
      m1.Gaussian_noise.variance.fix(0.1)
      GP_models.append(m1)


  round_holder= []    
  reward_holder= []    
  
  for i in range(K):
    round_holder.append([])
    reward_holder.append([])

  regret_holder=np.zeros(T)

  for t in range(T):
    sample_TS=np.zeros(K)

    for i in range(K):
      X =  np.array( (t)).reshape(-1,1) 
      sample_TS[i] = float(GP_models[i].posterior_samples_f(X,size=1).reshape(-1,1))

    next_pull=np.argmax(sample_TS) #determine the next arm to pull

    if t >=1:
      if (next_pull != choice[-1]) : #check whether we switch the arm at time t
        switch = 1
      else:
        switch = 0
    else:
      switch = 0

    regret=max_reward[t]-arms[next_pull][t]#calculate the regret
    regret_holder[t]=regret+C*switch

    reward=reward_generator(t,arms[next_pull],sigma=0.1) #obtain the reward

    round_holder[next_pull].append(t)
    reward_holder[next_pull].append(reward)
    choice.append(next_pull)

    X_sample= np.array(round_holder[next_pull]).reshape(-1,1)     
    Y_sample=np.array(reward_holder[next_pull]).reshape(-1,1)
    
    if model == 'RBF':
      kernel = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)
    elif model == 'Matern52':
      kernel = GPy.kern.Matern52(input_dim=1)
    
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(0.1)
    m.optimize()

    GP_models[next_pull]=m

  return regret_holder,choice




def padding(t,ti,sigma_i,sigma=0.1):
  return sigma*math.sqrt(8*math.log(t+1)/ti)+sigma_i*math.sqrt(8*t*math.log(t+1))


def UCB_f(T,*arms):

  choice = []
  
  max_reward = np.maximum.reduce(arms)
  K =len(arms) 

  regret_holder=np.zeros(T)
  
  mu_holder=[]
  N_holder=[]
  sigma_holder=[]
  reward_holder=[]
  round_holder=[]

  for i in range(K):
    mu_holder.append(0.)
    N_holder.append(1.)
    sigma_holder.append(1)
    reward_holder.append([])
    round_holder.append([])


  for t in range(T):
    
    UCB_holder=np.zeros(K)
    
    for i in range(K):
      UCB_holder[i]=mu_holder[i]+padding(t,N_holder[i],sigma_holder[i],0.1)

    next_pull=np.argmax(UCB_holder) #determine the next arm to pull
    regret=max_reward[t]-arms[next_pull][t]#calculate the regret
    regret_holder[t]=regret

    reward=reward_generator(t,arms[next_pull],sigma=0.1) #obtain the reward
    reward_holder[next_pull].append(reward)

    mu_holder[next_pull]=np.sum(np.array(reward_holder[next_pull]))
    N_holder[next_pull]=N_holder[next_pull]+1

    round_holder[next_pull].append(t)
    
    sigma_holder[next_pull]=math.sqrt(BM_calibration(np.array(round_holder[next_pull]),np.array(reward_holder[next_pull]),sigma=0.1))
    choice.append(next_pull)

  return regret_holder





def DP_pipeline(pre1, pre2, C, current_pos): 

  # there are two product lines: 0 and 1
  # the switching cost is C
  # we should find the best path
  if len(pre1)>=2:
  
    if current_pos==0:
      e1 = 0
      e2 = -C
    else:
      e1 = -C
      e2 = 0

    f1 = e1+pre1[0]
    f2 = e2+pre2[0]
    
    line1 = []
    line2 = []


    for i in range(1,len(pre1)):

      f1_old = f1
      f2_old = f2

      #update value for f1
      if f1_old+pre1[i]>= f2_old+pre1[i]-C:
        f1 = f1_old+pre1[i]
        line1.append(0)
      else:
        f1 = f2_old+pre1[i]-C
        line1.append(1)

      if f2_old+pre2[i]>= f1_old+pre2[i]-C:
        f2 = f2_old+pre2[i]
        line2.append(1)
      else:
        f2 = f1_old+pre2[i]-C
        line2.append(0)


      f_best = max(f1,f2)
      if f1>f2:
        final_pos = 0
      else:
        final_pos = 1


    line_res = [final_pos]
    state = final_pos
    for i in range(len(pre1)-1,0,-1):

      if state == 0:
        line_res.append(line1[i-1])
        state=line1[i-2]
      else:
        line_res.append(line2[i-1])
        state=line2[i-2]

    return line_res[::-1]


  elif len(pre1)==1:

    if current_pos==0:
      e1 = 0
      e2 = -C
    else:
      e1 = -C
      e2 = 0

    f1 = e1+pre1[0]
    f2 = e2+pre2[0]

    if f1>f2:
      line_res=[0]
    else:
      line_res=[1]

  return line_res




def GPR_DP(T,C, step_control, arm1,arm2,lower_bound=5,):

  choice = []
  arms=[arm1,arm2]
  max_reward = np.maximum.reduce(arms)
  K = 2

  X_sample=np.array([[0.0]])
  Y_sample=np.array([[0.0]])

  GP_models = []

  lengthscale=6
  variance=3

  for i in range (K):
    kernel1 = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)   #using RFP kernel
    m1 = GPy.models.GPRegression(X_sample,Y_sample,kernel1)
    m1.Gaussian_noise.variance.fix(0.1)
    GP_models.append(m1)

  l1 = float(GP_models[0].rbf.lengthscale)
  l2 = float(GP_models[1].rbf.lengthscale)

  round_holder= []    
  reward_holder= []    
  
  for i in range(2):
    round_holder.append([])
    reward_holder.append([])

  regret_holder=np.zeros(T)

  for t in range(T):

    l1 = float(GP_models[0].rbf.lengthscale)
    l2 = float(GP_models[1].rbf.lengthscale)

    sample_TS=[]
    future_step = int(min(max(lower_bound,l1),max(lower_bound,l2)))+step_control

    for i in range(K):
      X = np.array(range(t,min(t+future_step,T+1))).reshape(-1,1) ##################
      #print(X[0])
      temp = GP_models[i].posterior_samples_f(X,size=1).reshape(-1)
      #print(temp)
      sample_TS.append(temp)

    if t==0:
      path = DP_pipeline(sample_TS[0], sample_TS[1], 0 , 1) #the first choice does not cost anything
      #print(path)
      next_pull = path[0]
    else:
      path = DP_pipeline(sample_TS[0], sample_TS[1], C , choice[-1]) #the first choice does not cost anything
      #print(path)
      next_pull = path[0]


    if t >=1 :

      if (next_pull != choice[-1]) : #check whether we switch the arm at time t
        switch = 1
      else:
        switch = 0
    else:
      switch = 0

    regret=max_reward[t]-arms[next_pull][t]#calculate the regret
    regret_holder[t]=regret+switch*C

    reward=reward_generator(t,arms[next_pull],sigma=0.1) #obtain the reward

    round_holder[next_pull].append(t)
    reward_holder[next_pull].append(reward)
    choice.append(next_pull)

    X_sample= np.array(round_holder[next_pull]).reshape(-1,1)     
    Y_sample=np.array(reward_holder[next_pull]).reshape(-1,1)
    

    kernel = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)
    m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
    m.Gaussian_noise.variance.fix(0.1)
    m.optimize()

    GP_models[next_pull]=m

  return regret_holder,choice,GP_models


def count_switch(path):

  switch = 0
  for i in range(1,len(path)):
    if path[i]!=path[i-1]:
      switch = switch+1

  return switch





def EF_plotting(T, *arms):

  rr = np.zeros(T)
  
  track_1=np.zeros(T)
  track_2=np.zeros(T)

  track_1_std=np.zeros(T)
  track_2_std=np.zeros(T)

  K = len(arms)
  max_reward = np.maximum.reduce(arms)
  choice = []

  gamma= 0.5

  mu_holder=[]
  sigma_holder=[]
  R_holder=[] #the time that arm i has not been pulled
  round_holder=[]
  reward_holder=[]

  for i in range(K):
    mu_holder.append(0.)
    sigma_holder.append(1.)
    R_holder.append(1) #the time that arm i has not been pulled
    round_holder.append([])
    reward_holder.append([])

  regret_holder=np.zeros(T)

  for t in range(T):
    
    track_1_std[t]=sigma_holder[0]*math.sqrt(R_holder[0])
    track_2_std[t]=sigma_holder[1]*math.sqrt(R_holder[1])

    sample_TS=np.zeros(K)
    
    for i in range(K):
      sample_TS[i]=float(np.random.normal(mu_holder[i],sigma_holder[i]*math.sqrt(R_holder[i]),1))

    next_pull=np.argmax(sample_TS) #determine the next arm to pull
    regret=max_reward[t]-arms[next_pull][t]#calculate the regret
    regret_holder[t]=regret

    

    

    reward=reward_generator(t,arms[next_pull],sigma=0.1) #obtain the reward
    mu_new=gamma**R_holder[next_pull]*mu_holder[next_pull]+(1-gamma**R_holder[next_pull])*reward
    mu_holder[next_pull]=mu_new

    round_holder[next_pull].append(t)
    reward_holder[next_pull].append(reward)
    choice.append(next_pull)



    if next_pull ==0:
      track_1[t]= reward
      if t >0:
        track_2[t]=  track_2[t-1]
      else:
        track_2[t]= 0

    if next_pull ==1:
      track_2[t]= reward
      if t >0:
        track_1[t]=  track_1[t-1]
      else:
        track_1[t]= 0



    for i in range(K):
      if i== next_pull:
        R_holder[i]=1
      else:
        R_holder[i]=R_holder[i]+1

    #print(R_holder[0])

    sigma_holder[next_pull]=math.sqrt(BM_calibration(np.array(round_holder[next_pull]),np.array(reward_holder[next_pull]),sigma=0.1))

    rr[t] = reward

   

    

  return track_1,track_2,track_1_std,track_2_std,rr