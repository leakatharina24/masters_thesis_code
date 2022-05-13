#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Lea Demelius
"""

import numpy as np
import os
# import matplotlib
# matplotlib.use('agg') #'tkagg')
from matplotlib import pyplot as plt
plt.close('all')

# %% load data

data = np.load('/home/demelius/experimental_data/v1_data.npz')

# %% extract variables

timeseries = data['timeseries'] #(39175 voxels, 266 time) 
depth = data['depth'] #(39175,) -> 0=white/grey matter interface, 1=CSF/grey matter interface
angle = data['angle'] #(39175,) -> 0° = right meridian. increase counter-clockwise
eccen = data['eccen'] #(39175,) -> 0° = central fixation
sigma = data['sigma'] #(39175,) -> voxel receptive field size
stimuli = data['stimuli'] #(768, 768, 266)
stim_xcoord = data['stim_xcoord'] #(768, 768)
stim_ycoord = data['stim_ycoord'] #(768, 768)

# %% compute cortical coordinates

x_vox = -np.cos(np.deg2rad(angle))*eccen
y_vox = -np.sin(np.deg2rad(angle))*eccen

# %% extract timeseries of interest

mask_of_interest = np.logical_and(depth>0, depth<1)
mask_of_interest2 = eccen <= 10 #5
ind_of_interest = np.where(np.logical_and(mask_of_interest, mask_of_interest2))[0]

timeseries_new = timeseries[ind_of_interest]
timeseries_reverse = timeseries[~ind_of_interest]
print(len(timeseries_new))
x_vox_new = x_vox[ind_of_interest]
y_vox_new = y_vox[ind_of_interest]
depth_new = depth[ind_of_interest]
sigma_new = sigma[ind_of_interest]

 # %% explore averaged timeseries
fontsize = 11.25
 
interv_x= 0.5
interv_y = 0.5
offset_x = 0
offset_y = -2.5
loc = 'down'

mask_of_interest4 = np.logical_and(x_vox_new>offset_x-interv_x, x_vox_new<offset_x+interv_x)
mask_of_interest5 = np.logical_and(y_vox_new>offset_y-interv_y, y_vox_new<offset_y+interv_y)
ind_of_interest2 = np.logical_and(mask_of_interest4, mask_of_interest5)

#print(len(ind_of_interest2))
#print(depth_new[ind_of_interest2])

center_timeseries = timeseries_new[ind_of_interest2,:]
print('Number of averaged timeseries: {}'.format(len(center_timeseries)))

mean_center_timeseries = np.mean(center_timeseries,axis=0)
#std_center_timeseries = np.std(center_timeseries,axis=0)/10
mean = np.mean(mean_center_timeseries)
std = np.std(mean_center_timeseries)

mean = np.mean(mean_center_timeseries[:5])
zscore_timeseries = (mean_center_timeseries - mean)/mean*100
#zscore_timeseries = (mean_center_timeseries - fmri_baseline)/fmri_baseline*100
#zscore_timeseries = mean_center_timeseries

x_max_vis_index = 76.7*(offset_x+interv_x)+383.5
x_min_vis_index = 76.7*(offset_x-interv_x)+383.5
y_max_vis_index = 76.7*(offset_y+interv_y)+383.5
y_min_vis_index = 76.7*(offset_y-interv_y)+383.5

start_t = 0
end_t = 266 #266 #133 #
time = range(start_t,end_t)
time_in_s = range(start_t*2, end_t*2,2)
averaged_stimulus = np.mean(stimuli[int(np.floor(y_min_vis_index)):int(np.floor(y_max_vis_index)),\
                              int(np.floor(x_min_vis_index)):int(np.floor(x_max_vis_index)),time],axis=(0,1))
    
x_max_vis_index = offset_x+interv_x
x_min_vis_index = offset_x-interv_x
y_max_vis_index = offset_y+interv_y
y_min_vis_index = offset_y-interv_y

### plot images separately
plt.figure(figsize=(3,3))
plt.imshow(stimuli[:,:,0], cmap='gray', origin='lower', extent=[-5,5,-5,5])
plt.title('exp. visual field', fontsize=fontsize*1.5)
# plt.plot(x_min_vis_index,y_max_vis_index,'ro',markersize=1)
# plt.plot(x_min_vis_index,y_min_vis_index,'ro',markersize=1)
# plt.plot(x_max_vis_index,y_max_vis_index,'ro',markersize=1)
# plt.plot(x_max_vis_index,y_min_vis_index,'ro',markersize=1)
rectangle = plt.Rectangle((x_min_vis_index,y_min_vis_index), x_max_vis_index-x_min_vis_index, y_max_vis_index-y_min_vis_index, fc='red',ec="red")
plt.gca().add_patch(rectangle)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks([-5,-2.5,0,2.5,5],['-5°','-2.5°','0°','2.5°','5°'])
plt.yticks([-5,-2.5,0,2.5,5],['-5°','-2.5°','0°','2.5°','5°'])
# plt.xticks([])
# plt.yticks([])

plt.figure(figsize=(4.5,3))
plt.plot(time_in_s,averaged_stimulus)
plt.xlabel('time in s', fontsize=fontsize*1.25)
plt.title('averaged stimulus', fontsize=fontsize*1.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.ylabel('intensity',fontsize=fontsize*1.25)

fontsize=10
plt.figure(figsize=(12,3))
plt.plot(time_in_s,zscore_timeseries[time],label='_nolegend_')
#plt.plot(time_in_s,zscore_timeseries[time],'ro', markersize=2,label='_nolegend_')
#plt.fill_between(time, mean_center_timeseries[time]-std_center_timeseries[time], \
#                mean_center_timeseries[time]+std_center_timeseries[time] ,alpha=0.3)

#path = '/calc/demelius/experimental_data/final_figures/figures_results'
#np.save(os.path.join(path, '8bars_{}/fMRI_exp.npy'.format(loc)), zscore_timeseries[time])

#plt.legend(['std/10'])
#plt.title('Averaged fMRI signal')
plt.title('experimental fMRI response', fontsize=fontsize*1.5)
plt.xlabel('time in s', fontsize=fontsize*1.25)
plt.ylabel('BOLD in $\Delta$%', fontsize=fontsize*1.25)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

### plot images in subplots
# plt.figure()
# plt.subplot(2,2,1)plt.xticks([])
#plt.yticks([])
# plt.imshow(stimuli[:,:,0], cmap='gray', origin='lower', extent=[-5,5,-5,5])
# plt.title('Location in visual field')
# # plt.plot(x_min_vis_index,y_max_vis_index,'ro',markersize=1)
# # plt.plot(x_min_vis_index,y_min_vis_index,'ro',markersize=1)
# # plt.plot(x_max_vis_index,y_max_vis_index,'ro',markersize=1)
# # plt.plot(x_max_vis_index,y_min_vis_index,'ro',markersize=1)
# rectangle = plt.Rectangle((x_min_vis_index,y_min_vis_index), x_max_vis_index-x_min_vis_index, y_max_vis_index-y_min_vis_index, fc='red',ec="red")
# plt.gca().add_patch(rectangle)
# # plt.xticks([])
# # plt.yticks([])
# plt.subplot(2,2,2)
# #plt.plot(time,stimuli[int(76.7*offset_x+383.5),int(76.7*offset_y+383.5),time])
# plt.plot(time_in_s,averaged_stimulus)
# plt.xlabel('time in s')
# plt.title('Averaged stimulus')
# plt.subplot(2,1,2)
# plt.plot(time_in_s,zscore_timeseries[time],label='_nolegend_')
# plt.plot(time_in_s,zscore_timeseries[time],'ro', markersize=2,label='_nolegend_')
# #plt.fill_between(time, mean_center_timeseries[time]-std_center_timeseries[time], \
# #                mean_center_timeseries[time]+std_center_timeseries[time] ,alpha=0.3)
# #plt.legend(['std/10'])
# plt.title('Averaged fMRI signal')
# plt.xlabel('time in s')
# plt.tight_layout()

#np.save('/home/demelius/experimental_data/averaged_stimulus_center_wide',averaged_stimulus)

# %% explore averaged timeseries layer-wise
fontsize = 10

interv_x= 0.5
interv_y = 0.5
offset_x = 0
offset_y =  -2.5
loc = 'down'
mask_of_interest4 = np.logical_and(x_vox_new>offset_x-interv_x, x_vox_new<offset_x+interv_x)
mask_of_interest5 = np.logical_and(y_vox_new>offset_y-interv_y, y_vox_new<offset_y+interv_y)

x_max_vis_index = 76.7*(offset_x+interv_x)+383.5
x_min_vis_index = 76.7*(offset_x-interv_x)+383.5
y_max_vis_index = 76.7*(offset_y+interv_y)+383.5
y_min_vis_index = 76.7*(offset_y-interv_y)+383.5

start_t = 0
end_t = 266
time = range(start_t,end_t)
time_in_s = range(start_t*2, end_t*2,2)
averaged_stimulus = np.mean(stimuli[int(np.floor(y_min_vis_index)):int(np.floor(y_max_vis_index)),\
                              int(np.floor(x_min_vis_index)):int(np.floor(x_max_vis_index)),time],axis=(0,1))

x_max_vis_index = offset_x+interv_x
x_min_vis_index = offset_x-interv_x
y_max_vis_index = offset_y+interv_y
y_min_vis_index = offset_y-interv_y


plt.figure(figsize=(3,3))
plt.imshow(stimuli[:,:,0], cmap='gray', origin='lower', extent=[-5,5,-5,5])
plt.title('Location of voxel \n in stimulated visual field')
# plt.plot(x_min_vis_index,y_max_vis_index,'ro',markersize=1)
# plt.plot(x_min_vis_index,y_min_vis_index,'ro',markersize=1)
# plt.plot(x_max_vis_index,y_max_vis_index,'ro',markersize=1)
# plt.plot(x_max_vis_index,y_min_vis_index,'ro',markersize=1)
rectangle = plt.Rectangle((x_min_vis_index,y_min_vis_index), x_max_vis_index-x_min_vis_index, y_max_vis_index-y_min_vis_index, fc='red',ec="red")
plt.gca().add_patch(rectangle)
# plt.xticks([])
# plt.yticks([])

plt.figure(figsize=(9,3))
#plt.subplot(2,2,2)
#plt.plot(time,stimuli[int(76.7*offset_x+383.5),int(76.7*offset_y+383.5),time])
plt.plot(time_in_s,averaged_stimulus)
plt.xlabel('time in s')
plt.title('Averaged stimulus')

plt.figure(figsize=(12,2))
    
for i in [2,1,0]:
    mask_of_interest6 = np.logical_and(depth_new>i/3, depth_new<(i+1)/3)
    ind_of_interest2 = np.where(np.logical_and(np.logical_and(mask_of_interest4, mask_of_interest5),mask_of_interest6))[0]
    
    print(len(ind_of_interest2))
    #print(depth_new[ind_of_interest2])
    
    if i == 0:
        layer_name = 'lower'
        color='gold'
    elif i == 1:
        layer_name = 'middle'
        color='red'
    else:
        layer_name = 'upper'
        color='blue'
    
    center_timeseries = timeseries_new[ind_of_interest2,:]
    
    mean_center_timeseries = np.mean(center_timeseries,axis=0)
    std_center_timeseries = np.std(center_timeseries,axis=0)/10
    mean = np.mean(mean_center_timeseries)
    std = np.std(mean_center_timeseries)
    #zscore_timeseries = (mean_center_timeseries - mean)/std
    mean = np.mean(mean_center_timeseries[:5])
    fmri = (mean_center_timeseries-mean)/mean*100
    
    
    #plt.subplot(2,1,2)
    plt.plot(time_in_s,fmri[time],color) #,label='_nolegend_')
    #plt.plot(time_in_s,zscore_timeseries[time],'ro', markersize=2,label='_nolegend_')
    #plt.xlabel('time in s')
    #plt.fill_between(time, mean_center_timeseries[time]-std_center_timeseries[time], \
    #                mean_center_timeseries[time]+std_center_timeseries[time] ,alpha=0.3)
    #plt.legend(['std/10'])
    
    # path = '/calc/demelius/experimental_data/final_figures/figures_results'
    # np.save(os.path.join(path, 'laminar_8bars_{}/fMRI_exp_{}.npy'.format(loc,layer_name)), zscore_timeseries[time])
    
plt.title('experimental laminar fMRI responses',fontsize=fontsize*1.5) #in {} layer'.format(layer_name))    
#plt.legend(['superficial','middle','deep'],loc='upper right', fontsize=fontsize*1.25)
plt.xlabel('time in s', fontsize=fontsize*1.25)
plt.ylabel('BOLD in $\Delta$%', fontsize=fontsize*1.25)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

#plt.tight_layout()

# %% plot neuron selection
interv_x= 0.5
interv_y = 0.5
offset_x = 0
offset_y = 0

fontsize = 25

x_max_vis_index = offset_x+interv_x
x_min_vis_index = offset_x-interv_x
y_max_vis_index = offset_y+interv_y
y_min_vis_index = offset_y-interv_y


plt.figure(figsize=(6,6))
plt.imshow(stimuli[:,:,0], cmap='gray', origin='lower', extent=[-5,5,-5,5])
plt.title('exp. visual field',fontsize=fontsize*1.5)
circle = plt.Circle((0,0),radius=5, fc='black', ec='yellow')
plt.gca().add_patch(circle)
rectangle = plt.Rectangle((x_min_vis_index,y_min_vis_index), x_max_vis_index-x_min_vis_index, y_max_vis_index-y_min_vis_index, fc='red',ec="red")
plt.gca().add_patch(rectangle)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks([-5,-2.5,0,2.5,5],['-5°','-2.5°','0°','2.5°','5°'])
plt.yticks([-5,-2.5,0,2.5,5],['-5°','-2.5°','0°','2.5°','5°'])

plt.figure(figsize=(12,6))
rectangle = plt.Rectangle((0,0), 240, 120, fc='black',ec="black")
plt.gca().add_patch(rectangle)
plt.xlim([0,240])
plt.ylim([0,120])
circle = plt.Circle((120,60),radius=30, fc='black', ec='yellow')
plt.gca().add_patch(circle)
rectangle = plt.Rectangle((120-3,60-3), 6, 6, fc='red',ec="red")
plt.gca().add_patch(rectangle)
plt.title('visual field of Billeh et al. model',fontsize=fontsize*1.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks([0,50,100,150,200],['0°','50°','100°','150°','200°'])
plt.yticks([0,20,40,60,80,100,120],['0°','20°','40°','60°','80°','100°','120°'])

plt.figure(figsize=(6,6))
circle = plt.Circle((0,0),radius=400, fc='black', ec='black')
plt.gca().add_patch(circle)
plt.xlim([-400,400])
plt.ylim([-400,400])
rectangle = plt.Rectangle((0-42,0-75), 84, 150, fc='red',ec="red")
plt.gca().add_patch(rectangle)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks([-400,0,400],['-400 $\mu$m','0 $\mu$m','400 $\mu$m'])
plt.yticks([-400,0,400],['-400 $\mu$m','0 $\mu$m','400 $\mu$m'])
plt.title('cortical plane',fontsize=fontsize*1.5)

# %% plot laminar balloon model results
path = '/home/demelius/experimental_data/final_figures/figures_results/matlab_files'
loc = 'down'

import scipy.io

bold_upper = scipy.io.loadmat(os.path.join(path, 'results_{}/laminarballoon_upper.mat'.format(loc)))['bold_up']
bold_middle = scipy.io.loadmat(os.path.join(path, 'results_{}/laminarballoon_middle.mat'.format(loc)))['bold_mid']
bold_lower = scipy.io.loadmat(os.path.join(path, 'results_{}/laminarballoon_lower.mat'.format(loc)))['bold_low']


fontsize = 10
t_bin = 100
time = len(bold_lower)*t_bin
time_axis = np.arange(0, time, t_bin)/1000
layer_names = ['superficial', 'middle','deep']

plt.figure(figsize=(12,2))
for bold3, color in zip([bold_upper, bold_middle, bold_lower],['blue','red','gold']):
    plt.plot(time_axis, bold3, color)
#plt.legend(layer_names,fontsize=fontsize*1.25,loc='upper right')
plt.xlabel('time in s',fontsize=fontsize*1.25)
plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
plt.title('laminar fMRI responses: Laminar balloon model',fontsize=fontsize*1.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# %% plot laminar model example
path = '/home/demelius/experimental_data/final_figures/figures_fMRImodels/matlab_files'

import scipy.io

bold_upper = scipy.io.loadmat(os.path.join(path, 'laminarballoon_upper.mat'))['bold_up'] 
bold_middle = scipy.io.loadmat(os.path.join(path, 'laminarballoon_middle.mat'))['bold_mid'] 
bold_lower = scipy.io.loadmat(os.path.join(path,'laminarballoon_lower.mat'))['bold_low'] 

fontsize = 11.25
t_bin = 100
time = len(bold_lower)*t_bin
time_axis = np.arange(0, time, t_bin)/1000
layer_names = ['superficial', 'middle','deep']

plt.figure(figsize=(6,3))
for bold3, color in zip([bold_upper, bold_middle, bold_lower],['blue','red','gold']):
    plt.plot(time_axis, bold3, color)
plt.legend(layer_names,fontsize=fontsize*1.25,loc='upper right')
plt.xlabel('time in s',fontsize=fontsize*1.25)
plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
plt.title('fMRI response',fontsize=fontsize*1.5)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)