#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import time



"""
6ac-1-deterministic-na
6ac-1-mixed-high
6ac-1-mixed-low
6ac-1-mixed-medium
6ac-1-stochastic-high
6ac-1-stochastic-low
6ac-1-stochastic-medium
6ac-10-deterministic-na
6ac-10-mixed-high
6ac-10-mixed-low
6ac-10-mixed-medium
6ac-10-stochastic-high
6ac-10-stochastic-low
6ac-10-stochastic-medium
6ac-100-deterministic-na
6ac-100-mixed-high
6ac-100-mixed-low
6ac-100-mixed-medium
6ac-100-stochastic-high
6ac-100-stochastic-low
6ac-100-stochastic-medium
6ac-1000-deterministic-na
6ac-1000-mixed-high
6ac-1000-mixed-low
6ac-1000-mixed-medium
6ac-1000-stochastic-high
6ac-1000-stochastic-low
6ac-1000-stochastic-medium

"""
all_folders = [
    "../data/Training/6ac-100-deterministic-na/",
    "../data/Training/6ac-100-mixed-low/",
    "../data/Training/6ac-100-mixed-medium/",
    "../data/Training/6ac-100-mixed-high/",
    "../data/Training/6ac-100-stochastic-low/",
    "../data/Training/6ac-100-stochastic-medium/",
    "../data/Training/6ac-100-stochastic-high/",
]
all_folders_temp = [
    "../data/Training/6ac-100-deterministic-na/",
    "../data/Training/6ac-100-mixed-medium/",
]





# In train_dqn_both_timesteps.ipynb:
MAX_TOTAL_TIMESTEPS = 10000
SEEDS = [42]
brute_force_flag = False
cross_val_flag = False
early_stopping_flag = False
CROSS_VAL_INTERVAL = 1000000
printing_intermediate_results = False

step = 0
for training_folder in all_folders_temp:
    step += 1
    print(f"Step {step}")
    TRAINING_FOLDERS_PATH = training_folder
    get_ipython().run_line_magic('run', 'train_dqn_both_timesteps.ipynb')
    get_ipython().run_line_magic('run', 'train_ppo_both.ipynb')


