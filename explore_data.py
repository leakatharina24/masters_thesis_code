# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 09:28:27 2021

@author: leade
"""
import numpy as np
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

# %% plot images of stimulus

for t in np.linspace(0,265,num=100,dtype=int):
    plt.figure()
    plt.imshow(stimuli[:,:,t],origin='lower', cmap='gray', extent=[-5, 5, -5, 5,])
    print(t)
    plt.show()
    
# %% Expore stimulus visual coordinates
print(np.max(stim_xcoord))
print(np.min(stim_xcoord))
print(np.all(np.all(stim_xcoord == stim_xcoord[0,:], axis = 0)))
print(np.max(stim_ycoord))
print(np.min(stim_ycoord))
print(np.all(np.all(stim_ycoord.T == stim_ycoord[:,0], axis = 0)))

# %% Plot stimulus
# import matplotlib
# matplotlib.use('tkagg')
# from matplotlib import pyplot as plt

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on (probably unnecessary)
fig.show() # Initially shows the figure

for i in range(np.shape(stimuli)[2]):
    viewer.clear() # Clears the previous image
    viewer.imshow(stimuli[:,:,i], extent=[-5, 5, 5, -5,], cmap='gray') # Loads the new image
    plt.pause(.008) # Delay in seconds
    fig.canvas.draw() # Draws the image to the screen
    
# %% resize stimulus

import cv2

index = 6
plt.figure()
plt.imshow(stimuli[:,:,index], origin='lower', cmap='gray', extent=[-5, 5, -5, 5,])
img_small = cv2.resize(stimuli[:,:,index], (10, 10))
plt.figure()
plt.imshow(img_small, origin='lower', cmap='gray', extent=[-5, 5, -5, 5,])

vid_small = np.zeros((10,10,260))
for index in range(260):
    vid_small[:,:,index] = cv2.resize(stimuli[:,:,index], (10,10))
 
# %% save resized stimulus
#np.save('C:/Users/leade/Documents/BME/Thesis/experimental_fMRIdata/downsized_stimuli',vid_small)

# %% upsample stimulus in time
upsampled_stimulus = np.repeat(vid_small, 2000, axis=2)
# plt.figure()
# plt.imshow(upsampled_stimulus[:,:,20001], cmap='gray', extent=[-5, 5, 5, -5,])
#only first bar
video = np.zeros((120,240,72000),dtype=int)
video[55:65,115:125,:] = upsampled_stimulus[:,:,:72000]*255
plt.figure()
plt.imshow(video[:,:,71000], cmap='gray')
video = np.swapaxes(video,0,2)
video = np.swapaxes(video,1,2)

#np.save('/home/demelius/experimental_data/final_stimuli_72s',video)

# %% upsample stimulus in time without making it that small
import cv2

index = 6
plt.figure()
plt.imshow(stimuli[:,:,index], origin='lower', cmap='gray', extent=[-5, 5, -5, 5,])
img_small = cv2.resize(stimuli[:,:,index], (60, 60))
plt.figure()
plt.imshow(img_small, origin='lower', cmap='gray', extent=[-5, 5, -5, 5,])

vid_midsize = np.zeros((60,60,260))
for index in range(260):
    vid_midsize[:,:,index] = cv2.resize(stimuli[:,:,index], (60,60))
    
# upsampled_stimulus = np.repeat(vid_midsize, 2000, axis=2)
# # plt.figure()
# # plt.imshow(upsampled_stimulus[:,:,20001], cmap='gray', extent=[-5, 5, 5, -5,])
# #only first bar
# video = np.zeros((120,240,72000),dtype=int)
# video[30:90,90:150,:] = upsampled_stimulus[:,:,:72000]*255
# plt.figure()
# plt.imshow(video[:,:,71000], cmap='gray')
# video = np.swapaxes(video,0,2)
# video = np.swapaxes(video,1,2)

#np.save('/home/demelius/experimental_data/big_stimuli_72s',video)

# %% generate stimulus 60x60

def generate_fMRIstimulus(t=0, diameter=60, stimulus_onset=10, time=1000):   
    jump = 0
    if (t >=26 and t<42) or (t>=154 and t<170) or (t>=282 and t<298) or (t>=410 and t<426):
        jump = 1
    elif (t>=42 and t<58) or (t>=170 and t<186) or (t>=298 and t<314) or (t>=426 and t<442):
        jump = 2
    elif (t>=58 and t<74) or (t>=186 and t<202) or (t>=314 and t<330) or (t>=442 and t<458):
        jump = 3
        
    gen_video = np.zeros((time,diameter+14,diameter+14))
    
    
    #bar 1
    if t>=stimulus_onset and t<74:
        i = t-stimulus_onset+jump
        gen_video[:,diameter+14-i-8:diameter+14-i,:] = np.ones((8,74))
    #bar 2
    elif t>=74 and t<138:
        i = t-74
        jump = int(i/2)
        print(jump)
        i = i+21+jump
        tt, yy, xx = np.meshgrid(range(time),range(diameter+14), range(diameter+14), indexing='ij')
        mask = np.logical_and(xx+yy>i,xx+yy<i+12)
        gen_video[mask] = 1
    #bar 3
    elif t>=138 and t<202:
        i = t-138+jump
        gen_video[:,:,diameter+14-i-8:diameter+14-i] = np.ones((74,8))
    #bar 4
    elif t>=202 and t<266:
        i = t-202
        jump = int(i/2)
        i = i+21+jump
        i = 62-i
        tt, yy, xx = np.meshgrid(range(time),range(diameter+14), range(diameter+14), indexing='ij')
        mask = np.logical_and(-xx+yy>i,-xx+yy<i+12)
        gen_video[mask] = 1
    #bar 5
    elif t>=266 and t<330:
        i = t-266+jump
        gen_video[:,i:i+8,:] = np.ones((8,74))
    #bar 6
    elif t>=330 and t<394:
        i = t-330
        jump = int(i/2)
        i = i+21+jump
        i = 135-i
        tt, yy, xx = np.meshgrid(range(time),range(diameter+14), range(diameter+14), indexing='ij')
        mask = np.logical_and(xx+yy>i,xx+yy<i+12)
        gen_video[mask] = 1
    #bar 7
    elif t>=394 and t<458:
        i = t-394+jump
        gen_video[:,:,i:i+8] = np.ones((74,8))
    #bar 8
    elif t>=458 and t<522:
        i = t-458
        jump = int(i/2)
        i = i+jump-52
        tt, yy, xx = np.meshgrid(range(time),range(diameter+14), range(diameter+14), indexing='ij')
        mask = np.logical_and(-xx+yy>i,-xx+yy<i+12)
        gen_video[mask] = 1
    elif t>=532:
        raise ValueError('Simulation too long. Not possible to use the fMRI stimulus generator')
            
        gen_video = gen_video[:,7:67,7:67]
        tt, yy, xx = np.meshgrid(range(time),range(diameter), range(diameter), indexing='ij')
        xx = xx-diameter/2+0.5
        yy = yy-diameter/2+0.5
        radius = np.sqrt(yy**2+xx**2)
        radius_mask = radius>diameter/2
        gen_video[radius_mask] = 0
        gen_video = np.flipud(gen_video)
        #gen_video = (gen_video-0.5)*2
        #video = -np.ones((time,120,240))
        video = np.zeros((time,120,240))
        video[:,int(60-diameter/2):int(60+diameter/2),int(120-diameter/2):int(120+diameter/2)] = gen_video
        video = np.expand_dims(video,axis=3)
        video = video.astype(np.float32)
        return video
 
# for t in range(0,72,2):      
#     gen_video = generate_fMRIstimulus(t=t,time=1)
#     plt.imshow(gen_video[0],origin='lower',cmap='gray')
#     plt.show()
#     plt.imshow(vid_midsize[:,:,int(t/2)].astype(int),origin='lower',cmap='gray')
#     plt.show()
   
t = 518 #458-522
gen_video = generate_fMRIstimulus(t=t,time=1)
plt.imshow(gen_video[0],cmap='gray')

plt.figure()
plt.imshow(np.round(vid_midsize[:,:,int(t/2)]),origin='lower', cmap='gray', extent=[0,60,60,0])

plt.figure()
plt.imshow(stimuli[:,:,int(t/2)],origin='lower', cmap='gray', extent=[-5, 5, -5, 5,])



# import matplotlib
# matplotlib.use('tkagg')
# from matplotlib import pyplot as plt

# fig = plt.figure()
# viewer = fig.add_subplot(111)
# plt.ion() # Turns interactive mode on (probably unnecessary)
# fig.show() # Initially shows the figure

# for t in range(532):
#     gen_video = generate_fMRIstimulus(t=t,time=1)
#     viewer.clear() # Clears the previous image
#     viewer.imshow(gen_video[0],cmap='gray') # Loads the new image
#     plt.pause(.008) # Delay in seconds
#     fig.canvas.draw()
# %% compare to other movie stimulus

comp_video = np.load('C:/Users/leade/Documents/BME/Thesis/natural_movies/natural_movie_one.30s.30fps.304x608.npy')
video_part1 = np.zeros((100, 120, 240, 1))
video_part2 = np.repeat(comp_video,33,axis=0)
comp_video = np.concatenate((video_part1,  video_part2[:20000,92:212,184:424,None]), axis=0)
comp_video = comp_video/255
print(np.min(comp_video))
print(np.max(comp_video))
# %% 
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion() # Turns interactive mode on (probably unnecessary)
fig.show() # Initially shows the figure

for i in range(np.shape(stimuli)[2]):
    viewer.clear() # Clears the previous image
    viewer.imshow(vid_small[:,:,i], extent=[-5, 5, 5, -5], cmap='gray') # Loads the new image
    plt.pause(.008) # Delay in seconds
    fig.canvas.draw() # Draws the image to the screen

# %% explore depth
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt

# plt.figure()

# plt.plot(depth)

print(max(depth))
print(min(depth))

mask_of_interest = np.logical_and(depth>0, depth<1)
print(len(np.where(mask_of_interest)[0]))

print(len(np.unique(depth[mask_of_interest])))

plt.figure()
plt.plot(np.sort(depth))
#plt.axis([0,len(depth),0,1])
# %% explore angle
print(max(angle))
print(min(angle))

# plt.figure()
# plt.plot(np.sort(angle))


# %% explore eccentricity
print(max(eccen))
print(min(eccen))

# plt.figure()
# plt.plot(np.sort(eccen))

# %% compute x,y visual coordinates

x_vox = -np.cos(np.deg2rad(angle))*eccen
y_vox = -np.sin(np.deg2rad(angle))*eccen

print(len(np.unique(x_vox)))
print(len(np.unique(y_vox)))


# fig = plt.figure()
# viewer = fig.add_subplot(111)
# plt.ion() # Turns interactive mode on (probably unnecessary)
# fig.show() # Initially shows the figure

# for i in range(37):
#     viewer.clear() # Clears the previous image
#     cntr2 = viewer.tricontourf(x_vox, y_vox, timeseries[:,i], cmap='gray') #, levels=14, cmap="RdBu_r")
#     plt.axis([-5,5,5,-5])
#     plt.pause(.008) # Delay in seconds
#     fig.canvas.draw() # Draws the image to the screen

# fig = plt.figure()
# viewer = fig.add_subplot(111)
# plt.ion() # Turns interactive mode on (probably unnecessary)
# fig.show() # Initially shows the figure

# for i in range(18,23):
#     viewer.clear() # Clears the previous image
#     viewer.scatter(x_vox, y_vox, s=0.5, c=timeseries[:,i]) #, levels=14, cmap="RdBu_r")
#     plt.axis([-5,5,-5,5])
#     plt.pause(.001) # Delay in seconds
#     fig.canvas.draw() # Draws the image to the screen
    
    
# plt.figure()
# cntr2 = plt.tricontourf(x_vox, y_vox, timeseries[:,6]) #, levels=14, cmap="RdBu_r")
# plt.colorbar(cntr2) #, ax=ax2)
# plt.axis([-5,5,5,-5])
#plt.plot(x_vox, y_vox, 'ko', ms=3)
# ax2.set(xlim=(-2, 2), ylim=(-2, 2))
# ax2.set_title('tricontour (%d points)' % npts)

# plt.subplots_adjust(hspace=0.5)
# plt.show()

# %% explore sigma
print(max(sigma))
print(min(sigma))

# plt.figure()
# plt.plot(np.sort(sigma))

plt.figure()
plt.scatter(eccen,sigma, s=0.5)
plt.xlabel('eccentricity')
plt.ylabel('sigma')

# %% extract timeseries of interest

mask_of_interest = np.logical_and(depth>0, depth<1)
mask_of_interest2 = eccen <= 5
ind_of_interest = np.where(np.logical_and(mask_of_interest, mask_of_interest2))[0]

timeseries_new = timeseries[ind_of_interest]
print(len(timeseries_new))
x_vox_new = x_vox[ind_of_interest]
y_vox_new = y_vox[ind_of_interest]
depth_new = depth[ind_of_interest]
sigma_new = sigma[ind_of_interest]

# %% explore single timeseries

print(np.max(timeseries))
print(np.min(timeseries))

print(np.max(timeseries_new))
print(np.min(timeseries_new))

index = 10
time = range(0,100)
plt.subplot(2,2,1)
plt.imshow(stimuli[:,:,20])
x_vis_index = 76.7*x_vox_new[index]+383.5
y_vis_index = 76.7*x_vox_new[index]+383.5
plt.title('Location of voxel \n in stimulated visual field')
plt.plot(x_vis_index,y_vis_index,'ro',markersize=2)
plt.subplot(2,2,2)
plt.plot(stimuli[round(y_vis_index),round(x_vis_index),time])
plt.title('Stimulus')
plt.subplot(2,1,2)
plt.plot(timeseries_new[index,time])
plt.plot(time,timeseries_new[index,time],'ro', markersize=2)
plt.title('fMRI signal')
plt.legend(['depth: {}'.format(round(depth_new[index],4))])
plt.tight_layout()

# %% explore averaged timeseries
interv_x= 0.5
interv_y = 0.5
offset_x = 0
offset_y = 2.5
mask_of_interest4 = np.logical_and(x_vox_new>offset_x-interv_x, x_vox_new<offset_x+interv_x)
mask_of_interest5 = np.logical_and(y_vox_new>offset_y-interv_y, y_vox_new<offset_y+interv_y)
ind_of_interest2 = np.logical_and(mask_of_interest4, mask_of_interest5)

#print(len(ind_of_interest2))
#print(depth_new[ind_of_interest2])

center_timeseries = timeseries_new[ind_of_interest2]

mean_center_timeseries = np.mean(center_timeseries,axis=0)
#std_center_timeseries = np.std(center_timeseries,axis=0)/10
mean = np.mean(mean_center_timeseries)
std = np.std(mean_center_timeseries)
zscore_timeseries = (mean_center_timeseries - mean)/std

x_max_vis_index = 76.7*(offset_x+interv_x)+383.5
x_min_vis_index = 76.7*(offset_x-interv_x)+383.5
y_max_vis_index = 76.7*(offset_y+interv_y)+383.5
y_min_vis_index = 76.7*(offset_y-interv_y)+383.5

start_t = 0
end_t = 36
time = range(start_t,end_t)
time_in_s = range(start_t*2, end_t*2,2)
averaged_stimulus = np.mean(stimuli[int(np.floor(y_min_vis_index)):int(np.floor(y_max_vis_index)),\
                              int(np.floor(x_min_vis_index)):int(np.floor(x_max_vis_index)),time],axis=(0,1))

plt.figure()
plt.subplot(2,2,1)
plt.imshow(stimuli[:,:,15], cmap='gray', origin='lower')
plt.title('Location of voxel \n in stimulated visual field')
plt.plot(x_min_vis_index,y_max_vis_index,'ro',markersize=1)
plt.plot(x_min_vis_index,y_min_vis_index,'ro',markersize=1)
plt.plot(x_max_vis_index,y_max_vis_index,'ro',markersize=1)
plt.plot(x_max_vis_index,y_min_vis_index,'ro',markersize=1)
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
#plt.plot(time,stimuli[int(76.7*offset_x+383.5),int(76.7*offset_y+383.5),time])
plt.plot(time_in_s,averaged_stimulus)
plt.title('Averaged stimulus')
plt.subplot(2,1,2)
plt.plot(time_in_s,zscore_timeseries[time],label='_nolegend_')
plt.plot(time_in_s,zscore_timeseries[time],'ro', markersize=2,label='_nolegend_')
#plt.fill_between(time, mean_center_timeseries[time]-std_center_timeseries[time], \
#                mean_center_timeseries[time]+std_center_timeseries[time] ,alpha=0.3)
#plt.legend(['std/10'])
plt.title('Averaged fMRI signal')
plt.tight_layout()

#np.save('/home/demelius/experimental_data/averaged_stimulus_center_wide',averaged_stimulus)

# %% explore averaged timeseries for 1 layer
interv_x= 0.5
interv_y = 0.5
offset_x = 0
offset_y = 0
mask_of_interest4 = np.logical_and(x_vox_new>offset_x-interv_x, x_vox_new<offset_x+interv_x)
mask_of_interest5 = np.logical_and(y_vox_new>offset_y-interv_y, y_vox_new<offset_y+interv_y)
mask_of_interest6 = np.logical_and(depth_new>1/3, depth_new<2/3)
ind_of_interest2 = np.where(np.logical_and(np.logical_and(mask_of_interest4, mask_of_interest5),mask_of_interest6))[0]

print(len(ind_of_interest2))
#print(depth_new[ind_of_interest2])

center_timeseries = timeseries_new[ind_of_interest2]

mean_center_timeseries = np.mean(center_timeseries,axis=0)
#std_center_timeseries = np.std(center_timeseries,axis=0)/10
mean = np.mean(mean_center_timeseries)
std = np.std(mean_center_timeseries)
zscore_timeseries = (mean_center_timeseries - mean)/std

x_max_vis_index = 76.7*(offset_x+interv_x)+383.5
x_min_vis_index = 76.7*(offset_x-interv_x)+383.5
y_max_vis_index = 76.7*(offset_y+interv_y)+383.5
y_min_vis_index = 76.7*(offset_y-interv_y)+383.5

time = range(0,37)
averaged_stimulus = np.mean(stimuli[int(np.floor(y_min_vis_index)):int(np.floor(y_max_vis_index)),\
                              int(np.floor(x_min_vis_index)):int(np.floor(x_max_vis_index)),time],axis=(0,1))

plt.figure()
plt.subplot(2,2,1)
plt.imshow(stimuli[:,:,15], cmap='gray', origin='lower')
plt.title('Location of voxel \n in stimulated visual field')
plt.plot(x_min_vis_index,y_max_vis_index,'ro',markersize=1)
plt.plot(x_min_vis_index,y_min_vis_index,'ro',markersize=1)
plt.plot(x_max_vis_index,y_max_vis_index,'ro',markersize=1)
plt.plot(x_max_vis_index,y_min_vis_index,'ro',markersize=1)
plt.xticks([])
plt.yticks([])
plt.subplot(2,2,2)
#plt.plot(time,stimuli[int(76.7*offset_x+383.5),int(76.7*offset_y+383.5),time])
plt.plot(time,averaged_stimulus)
plt.title('Averaged stimulus')
plt.subplot(2,1,2)
plt.plot(time,zscore_timeseries[time],label='_nolegend_')
plt.plot(time,zscore_timeseries[time],'ro', markersize=2,label='_nolegend_')
#plt.fill_between(time, mean_center_timeseries[time]-std_center_timeseries[time], \
#                mean_center_timeseries[time]+std_center_timeseries[time] ,alpha=0.3)
#plt.legend(['std/10'])
plt.title('Averaged fMRI signal in upper layer')
plt.tight_layout()

# %%
#plt.figure()
# for ind in ind_of_interest:
#     plt.plot(timeseries[ind,:])
#     plt.plot(t_ind,timeseries[ind,t_ind],'ro')

# print(t_ind[0])
# plt.figure()
# plt.plot(range(12,20),timeseries[ind_of_interest[0],12:20])
# plt.plot(t_ind[0][:4],timeseries[ind_of_interest[0],t_ind[0][:4]],'ro')

# print(t_ind[0][4:9])
# plt.figure()
# plt.plot(range(56,63),timeseries[ind_of_interest[0],56:63])
# plt.plot(t_ind[0][4:8],timeseries[ind_of_interest[0],t_ind[0][4:8]],'ro')


# %%
plt.figure()
time = range(0,10)
plt.plot(time,np.mean(timeseries[:,time],axis=0))
# %% read json file

import json
with open('C:/Users/leade/Documents/BME/Thesis/experimental_fMRIdata/sub-GVW19_task-retmap-bars_run-01_bold.json','r') as json_file:
    info = json.load(json_file)

# %% explore json file
print(info.keys())
print(info['EchoTime'])
print(info['TaskName'])
print(info['TotalReadoutTime'])