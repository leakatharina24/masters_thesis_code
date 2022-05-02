#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 07:59:51 2022

@author: demelius
"""

import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats


path = '/calc/demelius/experimental_data/final_figures/figures_results'

# %% cross crorrelations: layer-unspecific

loc = 'down'
fmri_exp = np.load(os.path.join(path, '8bars_{}/fMRI_exp.npy'.format(loc)))
fr_pred = np.load(os.path.join('/calc/demelius/spiketrains_for_fMRI/MuckliStim_532s_0bis1_spikes','fr_all_{}.npy'.format(loc)))
fr_pred = [(a + b) / 2 for a, b in zip(fr_pred[::2], fr_pred[1::2])]
fmri_ltm = np.load(os.path.join(path, '8bars_{}/LTM.npy'.format(loc)))
fmri_mech = np.load(os.path.join(path, '8bars_{}/Mechanistic.npy'.format(loc)))
fmri_balloon = np.load(os.path.join(path, '8bars_{}/Balloon.npy'.format(loc)))
df = pd.DataFrame({'firingrate':fr_pred,'fMRI_exp':fmri_exp,'ltm':fmri_ltm,'mechanistic':fmri_mech,'balloon':fmri_balloon}) 
time = 532


### -1 = negatively correlated, 0 = not correlated, 1 = perfectly correlated
r, p = stats.pearsonr(df.dropna()['firingrate'], df.dropna()['fMRI_exp'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
r, p = stats.pearsonr(df.dropna()['fMRI_exp'], df.dropna()['ltm'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
r, p = stats.pearsonr(df.dropna()['fMRI_exp'], df.dropna()['balloon'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
r, p = stats.pearsonr(df.dropna()['fMRI_exp'], df.dropna()['mechanistic'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


def plot_crosscorr(d1, label1, d2, label2):
    fontsize=11.25
    seconds = 10
    #a = [print(lag)for lag in range(-int(seconds/2),int(seconds/2+1))]
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds/2),int(seconds/2+1))]
    offset = (np.floor(len(rs)/2)-np.argmax(rs))*2
    plt.figure(figsize=(6,3))
    plt.plot(rs)
    plt.axvline(np.floor(len(rs)/2),color='k',linestyle='--',label='Center')
    plt.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    #plt.axhline(np.max(rs),color='b',linestyle='--',label='Max Pearson')
    print(np.max(rs))
    print(offset)
    plt.title('<>',fontsize=fontsize*1.5)
    plt.title('{} leads'.format(label2),loc='right',fontsize=fontsize*1.25)
    plt.title('{} leads'.format(label1),loc='left',fontsize=fontsize*1.25)
    #plt.xticks([0,2.5,6,8.5,11],[-10,-5,0,5,10])
    plt.xticks([0,2.5,5,7.5,10],[-10,-5,0,5,10],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Offset in s',fontsize=fontsize*1.25)
    plt.ylabel('Pearson r',fontsize=fontsize*1.25)
    plt.legend(fontsize=fontsize*1.2,loc='lower right')
    
d1 = df['firingrate']
d2 = df['fMRI_exp']
plot_crosscorr(d1,'firing rate',d2,'exp. fMRI')
# plot_crosscorr(df['firingrate'],df['ltm'])
# plot_crosscorr(df['firingrate'],df['balloon'])
# plot_crosscorr(df['firingrate'],df['mechanistic'])
plot_crosscorr(df['ltm'], 'LTM',df['fMRI_exp'],'exp. fMRI')
plot_crosscorr(df['balloon'],'balloon m.',df['fMRI_exp'],'exp.fMRI')
plot_crosscorr(df['mechanistic'],'mechanistic m.',df['fMRI_exp'],'exp. fMRI')

from statsmodels.tsa.stattools import grangercausalitytests

#perform Granger-Causality test
maxlag = 6
granger_dict = grangercausalitytests(df[['fMRI_exp','firingrate']], maxlag=maxlag,verbose=False)
for i in range(1,maxlag+1):
    print(granger_dict[i][0]['params_ftest'])
#granger_dict = grangercausalitytests(df[['firingrate','fMRI_exp']], maxlag=6) 

# %% cross correlation: laminar
path = '/calc/demelius/experimental_data/final_figures/figures_results'
loc = 'up'
df_dict = {}
for layer in ['upper', 'middle', 'lower']:
    df_dict['fmri_exp_{}'.format(layer)] = np.load(os.path.join(path, 'laminar_8bars_{}/fMRI_exp_{}.npy'.format(loc,layer)))
    temp = np.load(os.path.join('/calc/demelius/spiketrains_for_fMRI/MuckliStim_532s_0bis1_spikes','fr_laminar_{}_tbin1000_{}.npy'.format(loc,layer)))
    df_dict['firingrate_{}'.format(layer)] = [(a + b) / 2 for a, b in zip(temp[::2], temp[1::2])]
df = pd.DataFrame(df_dict) #, 'fMRI_pred':fmri_pred})
time = 532


### -1 = negatively correlated, 0 = not correlated, 1 = perfectly correlated
for layer in ['upper', 'middle', 'lower']:
    r, p = stats.pearsonr(df.dropna()['firingrate_{}'.format(layer)], df.dropna()['fmri_exp_{}'.format(layer)])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


def plot_crosscorr(d1, label1, d2, label2):
    fontsize=11.25
    seconds = 10
    #a = [print(lag)for lag in range(-int(seconds/2),int(seconds/2+1))]
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds/2),int(seconds/2+1))]
    offset = (np.floor(len(rs)/2)-np.argmax(rs))*2
    plt.figure(figsize=(6,3))
    plt.plot(rs)
    plt.axvline(np.floor(len(rs)/2),color='k',linestyle='--',label='Center')
    plt.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    #plt.axhline(np.max(rs),color='b',linestyle='--',label='Max Pearson')
    print(np.max(rs))
    print(offset)
    plt.title('<>',fontsize=fontsize*1.5)
    plt.title('{} leads'.format(label2),loc='right',fontsize=fontsize*1.25)
    plt.title('{} leads'.format(label1),loc='left',fontsize=fontsize*1.25)
    #plt.xticks([0,2.5,6,8.5,11],[-10,-5,0,5,10])
    plt.xticks([0,2.5,5,7.5,10],[-10,-5,0,5,10],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Offset in s',fontsize=fontsize*1.25)
    plt.ylabel('Pearson r',fontsize=fontsize*1.25)
    plt.legend(fontsize=fontsize*1.2,loc='lower right')

for layer in ['upper', 'middle', 'lower']:
    plot_crosscorr(df['firingrate_{}'.format(layer)], 'firing rate',df['fmri_exp_{}'.format(layer)],'exp. fMRI')


from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 6
for layer in ['upper', 'middle', 'lower']:
    granger_dict = grangercausalitytests(df[['fmri_exp_{}'.format(layer),'firingrate_{}'.format(layer)]], maxlag=maxlag,verbose=False)
    for i in range(1,maxlag+1):
        print(granger_dict[i][0]['params_ftest'])
    print('--------------------')



