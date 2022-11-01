# 😺😺😺
# Non-stationary multi-armed bandit probelms

## Abstract

One assumption of multi-armed bandit problems is that the reward distribution is stationary. How-
ever, in real-life problems, the reward distribution can be non-stationary, i.e. the parameter of the
distribution can change with time. This paper handles non-stationary multi-armed bandit problems
by modelling the bandit processes as Gaussian processes to balance exploration and exploitation.
Further, we introduce a switching cost into this problem to make the problem more practical but at
the same time, more challenging. In order to handle the switching cost, Gaussian process regression
along with dynamic programming is utilized. The empirical results show that we should reduce
exploration when the switching cost is high.

## Installation
Our python version is 3.9.7.
```bash
git clone https://github.com/HanyangHenry-Wang/restless-bandit.git && cd restless-bandit
pip install -r requirements.txt --upgrade
pip install -e .
```


## Data and figures
The experiments are done in experiment.ipynb and new_experiments.ipynb. The results of experiments are stored in "result" and "new result" file.
