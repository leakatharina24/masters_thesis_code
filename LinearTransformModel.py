# -*- coding: utf-8 -*-
"""
Model: Boynton et al., “Linear systems analysis of the fmri signal,” Neuroimage, vol. 62, no. 2, pp. 975–984, 2012.
Code: Lea Demelius
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy

class LinearTransformModel:
    def __init__(self,time_axis, stimulus):
        self.time_axis = time_axis
        self.stimulus = stimulus
              
    def hemodynamic_response(self, t, u, n = 3, tau = 1.5):
        #impulse response
        h = ((t/tau)**(n-1)*np.exp(-t/tau))/(tau*scipy.special.gamma(n)) 
        #h = scipy.stats.gamma.pdf(t, n, loc=0, scale = tau)
    
        #Hemodynamic response
        dt = t[1]-t[0]
        hemo = dt*np.convolve(u,h)
    
        return hemo
    
    def solve(self, plot=True, save_plot=False, name=''):
        response = self.hemodynamic_response(self.time_axis, self.stimulus)
        
        if plot:
            fontsize=11.25
            plt.figure(figsize=(6,3))
            plt.plot(self.time_axis,response[:len(self.time_axis)]/90*100)
            plt.xlabel('time in s',fontsize=fontsize*1.25)
            plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
            plt.title('fMRI response',fontsize=fontsize*1.5)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if save_plot:
                plt.savefig('Model1_{}.png'.format(name))
                
        return response[:len(self.time_axis)]/90*100
        
        