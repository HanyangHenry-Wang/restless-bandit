from restless_bandit.algorithms import * 
from restless_bandit.arm_generator import *
from run_experiment.exp_discount import experiment2
import sys

def run(exp_num,C):
    
    if exp_num == 2:
        experiment2(C)
       
        



if __name__=='__main__':
    
    exp_num = sys.argv[1]
    C = sys.argv[2]
    
    run(exp_num,C)

