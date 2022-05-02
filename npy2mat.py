#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 08:37:12 2021

@author: demelius
"""

import scipy.io
import numpy as np
import os

# def average_spikes(z, time, t_bin=100):
#     temp_avg = np.count_nonzero(z[:t_bin,:],axis=0)
#     for i in range(1,int(time/t_bin)):
#         temp_avg = np.vstack((temp_avg, np.count_nonzero(z[i*t_bin:(i+1)*t_bin,:],axis=0)))
#     firing_rate = np.mean(temp_avg, axis=1) #population rate
#     return firing_rate

t_bin = 100
loc = 'up'
path = '/home/demelius/spiketrains_for_fMRI/MuckliStim_532s_0bis1_spikes'
for file in ['fr_laminar_{}_tbin{}_upper.npy'.format(loc,t_bin),'fr_laminar_{}_tbin{}_middle.npy'.format(loc,t_bin),'fr_laminar_{}_tbin{}_lower.npy'.format(loc,t_bin)]: #['firingrate_tbin10_lower.npy', 'firingrate_tbin10_middle.npy', 'firingrate_tbin10_upper.npy']:
    fr = np.load(os.path.join(path, file))
    scipy.io.savemat(os.path.join(path, file[:-3]+'mat'),{'firingrate':fr,'t_bin':t_bin, 'loc':loc})