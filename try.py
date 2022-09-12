from algorithm.new_algorithms import * 
from bandit_process.new_arms import *
from matplotlib import pyplot as plt
path0 = [0,5,5,0,0,0]
path1 =  [1,0,0,10,10,10]
path2 = [2,0,0,4,0,0]
res = Assembly_Line_Scheduling(5,0,[path0,path1,path2])
print(res)
# a=GP_Arm(lengthscale=10, variance=5,sigma=0.1, T=200)
# temp1 = a.arm_path
# rewards=a.reward_path()
# plt.scatter(200,rewards)
# plt.show()