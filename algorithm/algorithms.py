import numpy as np
import matplotlib;matplotlib.rcParams['figure.figsize'] = (8,5)
import GPy
import math
from bandit_process.arm_generator import reward_generator



def BM_calibration(round_holder,reward_holder,sigma=0.1):
  
  """this function is used to calibration \sigma_i in UCB_f and DTS

  Args:
      round_holder (array): the array records the time that arm i is pulled
      reward_holder (array): the array records the reward of arm i 
      sigma (float, optional): the std of reward distribution. Defaults to 0.1.

  Returns:
      float: sigma^2
  """

  kernel = GPy.kern.Brownian(input_dim=1)
  m = GPy.models.GPRegression(round_holder.reshape(-1,1),reward_holder.reshape(-1,1),kernel)
  m.Gaussian_noise.variance.fix(sigma)
  m.optimize()

  var=m.Brownian.variance[0]

  return var



def DTS(T, *arms):
  """dynamic thompson sampling

  Args:
      T (int): number of rounds

  Returns:
      list: regret at each term 
  """

  K = len(arms)
  max_reward = np.maximum.reduce(arms)
  choice = []

  gamma= 0.5

  mu_holder=[]
  sigma_holder=[]
  R_holder=[] 
  round_holder=[]
  reward_holder=[]

  for i in range(K):
    mu_holder.append(0.)
    sigma_holder.append(1.)
    R_holder.append(1) 
    round_holder.append([])
    reward_holder.append([])

  regret_holder=np.zeros(T)

  for t in range(T):

    sample_TS=np.zeros(K)
    
    for i in range(K):
      sample_TS[i]=float(np.random.normal(mu_holder[i],sigma_holder[i]*math.sqrt(R_holder[i]),1))

    next_pull=np.argmax(sample_TS)
    regret=max_reward[t]-arms[next_pull][t]
    regret_holder[t]=regret
    

    reward=reward_generator(t,arms[next_pull],sigma=0.1) 
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
  """_summary_

  Args:
      T (_type_): _description_
      model (_type_): _description_
      C (_type_): _description_

  Returns:
      _type_: _description_
  """  

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




def padding(t,ti,sigma_i,sigma=0.1):   #this is padding function of UCB_f
  return sigma*math.sqrt(8*math.log(t+1)/ti)+sigma_i*math.sqrt(8*t*math.log(t+1))


def UCB_f(T,*arms):
  """_summary_

  Args:
      T (_type_): _description_

  Returns:
      _type_: _description_
  """  

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
  
  """this function calculate the strategy in Assembly Line Scheduling problem

  Args:
      pre1 (array or list): the future path of arm 1 
      pre2 (array or list): the future path of arm 2
      C (float): switching cost
      current_pos (int): 0 (arm 1) or 1 (arm 2)

  Returns:
       list: the strategy
  """  

  
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





def GPR_DP(T,C, arm1,arm2,discount_factor=1,TS=True):

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
    kernel1 = GPy.kern.RBF(input_dim=1,variance=variance,lengthscale=lengthscale)   #using RBF kernel
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
    future_step = int(min(max(5,l1),max(5,l2)))
    

    for i in range(K):
      X = np.array(range(t,min(t+future_step,T+1))).reshape(-1,1) 
    
      if TS is True:
        temp = GP_models[i].posterior_samples_f(X,size=1).reshape(-1)
       
      else:
        temp,_ = GP_models[i].predict(X)
        temp = temp.reshape(-1)
        
     
      
      df=np.ones(len(temp))
      for j in range(len(temp)):
        df[j]=discount_factor**j
      
     
      
      sample_TS.append(df*temp)

    if t==0:
      path = DP_pipeline(sample_TS[0], sample_TS[1], 0 , 1) #the first choice does not cost anything
      next_pull = path[0]
    else:
      path = DP_pipeline(sample_TS[0], sample_TS[1], C , choice[-1]) #the first choice does not cost anything  
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

  return regret_holder,choice


def count_switch(path):

  switch = 0
  
  for i in range(1,len(path)):
    if path[i]!=path[i-1]:
      switch = switch+1

  return switch