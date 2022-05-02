# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:29:29 2021

@author: leade
"""
import numpy as np
from matplotlib import pyplot as plt
#import rungekutte
#import rungekutte_double
from scipy.interpolate import interp1d
from scipy.special import gamma
#import pdb
from scipy.integrate import odeint

class BalloonModel:
    """
    Represents the Balloon model from Buxton, Richard B., et al. "Modeling the hemodynamic response to brain activation." Neuroimage 23 (2004): S220-S233.
    

    Attributes:
    ----------
    time_axis : numpy array
        time for which the model should be simulated
    stimulus: numpy array
    tau0 : float
        transit time through the balloon
    E0 : float
        baseline oxygen extraction fraction
    VO : float
        baseline blood volume
    alpha : float
        steady-state flow-volume-relation
    outflow_linear : boolean
        can be used to switch to linear flow-volume relation
    tau : float
        viscoelastic time constant of balloon
    """
    def __init__(self, time_axis, stimulus, tau0 = 3.0, E0 = 0.4, V0 = 0.03, alpha = 0.4, outflow_linear = True, tau = 20.):
        self.tau0 = tau0
        self.E0 = E0
        self.V0 = V0
        self.alpha = alpha
        
        self.outflow_linear = outflow_linear
        self.tau = tau
        
        self.time_axis = time_axis
        self.stimulus = stimulus
      
    def linearfunc(self, x, x1, y1, x2, y2):
        k = (y2-y1)/(x2-x1)
        d = y1 - k*x1
        return x*k+d
       
    def f_out(self, v, t, f_in):
        if self.outflow_linear:
            return self.linearfunc(v,1,1,1.3,1.7)
        else:
            result = (self.tau0*v**(1/self.alpha)+self.tau*f_in)/(self.tau0+self.tau)
            return result
        
    def dfdt(self,y,t,params):
        f_in_func, cmro2_func = params
        v, q = y
        dt_v = 1/self.tau0*(f_in_func(t)-self.f_out(v,t, f_in_func(t)))
        dt_q = 1/self.tau0*(cmro2_func(t)-self.f_out(v,t, f_in_func(t))*q/v)
        
        return [dt_v, dt_q]
        
    def h(self, tau, delay=None):
        k = 3
        tau_h = 0.242*tau
        if delay is None:
            return ((self.time_axis/tau_h)**k*np.exp(-self.time_axis/tau_h))/(k*tau_h*gamma(k))
        else:
            return ((self.time_axis-delay/tau_h)**k*np.exp(-(self.time_axis-delay)/tau_h))/(k*tau_h*gamma(k))
        
    def CBF_response(self, tau_f = 4., f1 = 1.5, delay = 0.5):
        h = self.h(tau_f, delay = delay)
        dt = self.time_axis[1]-self.time_axis[0]
        f_in = 1. + (f1 -1)*dt*np.convolve(h, self.stimulus)
        f_in = f_in[:len(self.time_axis)]
        return f_in
        
    def CMRO2_response(self, tau_m = 4., f1 = 1.5, n = 3):
        h = self.h(tau_m)
        dt = self.time_axis[1]-self.time_axis[0]
        m = 1. + (f1-1)/n*dt*np.convolve(h, self.stimulus)
        m = m[:len(self.time_axis)]
        return m
    
    def solve(self, plot=True, save_plot=False, name=''):
        f_in = self.CBF_response()
        cmro2 = self.CMRO2_response()
        
        f_in_func = interp1d(self.time_axis,f_in,bounds_error=False, fill_value="extrapolate")
        cmro2_func = interp1d(self.time_axis,cmro2,bounds_error=False, fill_value="extrapolate")
        
        params = [f_in_func, cmro2_func]
        solver = odeint(self.dfdt, [1.0, 1.0], self.time_axis, args=(params,))
        v = solver[:,0]
        q = solver[:,1]
        
        a1 = 3.4
        a2 = 1.0
        bold = self.V0*(a1*(1-q)-a2*(1-v))
        
        # A=0.075
        # beta=1.5
        # bold2 = A*(1-v**(1-beta/self.alpha)*cmro2**beta)
        
        if plot:
            fontsize=11.25
            fig, axs = plt.subplots(2,3,figsize=(12,6))
            axs[0,0].plot(v,self.f_out(v, self.time_axis, f_in))
            axs[0,0].set_title("outflow",fontsize=fontsize*1.5)
            axs[0,0].set_xlabel('volume v',fontsize=fontsize*1.25)
            axs[0,0].set_ylabel('$f_{out}(v)$',fontsize=fontsize*1.25)
            axs[0,0].xaxis.set_tick_params(labelsize=fontsize)
            axs[0,0].yaxis.set_tick_params(labelsize=fontsize)
            axs[0,1].plot(self.time_axis,f_in)
            axs[0,1].plot(self.time_axis, self.f_out(v, self.time_axis, f_in))
            axs[0,1].set_title('blood flow',fontsize=fontsize*1.5)
            axs[0,1].legend(['$f_{in}(t)$', '$f_{out}(t)$'],loc='upper right',fontsize=fontsize*1.25) #,fontsize='x-small')
            axs[0,1].set_xlabel('time in s',fontsize=fontsize*1.25)
            axs[0,1].xaxis.set_tick_params(labelsize=fontsize)
            axs[0,1].yaxis.set_tick_params(labelsize=fontsize)
            axs[0,2].plot(self.time_axis, cmro2)
            axs[0,2].set_xlabel('time in s',fontsize=fontsize*1.25)
            axs[0,2].set_ylabel('m(t)',fontsize=fontsize*1.25)
            axs[0,2].set_title('CMRO2',fontsize=fontsize*1.5)
            axs[0,2].xaxis.set_tick_params(labelsize=fontsize)
            axs[0,2].yaxis.set_tick_params(labelsize=fontsize)
            axs[1,0].plot(self.time_axis,v)
            axs[1,0].set_title('volume',fontsize=fontsize*1.5)
            axs[1,0].set_ylabel("v(t)",fontsize=fontsize*1.25)
            axs[1,0].set_xlabel('time in s',fontsize=fontsize*1.25)
            axs[1,0].xaxis.set_tick_params(labelsize=fontsize)
            axs[1,0].yaxis.set_tick_params(labelsize=fontsize)
            axs[1,1].plot(self.time_axis,q)
            axs[1,1].set_title('dHb',fontsize=fontsize*1.5)
            axs[1,1].set_ylabel("q(t)",fontsize=fontsize*1.25)
            axs[1,1].set_xlabel('time in s',fontsize=fontsize*1.25)
            axs[1,1].xaxis.set_tick_params(labelsize=fontsize)
            axs[1,1].yaxis.set_tick_params(labelsize=fontsize)
            axs[1,2].plot(self.time_axis,100*bold) #self.bold/max(self.bold)
            axs[1,2].set_title('fMRI response',fontsize=fontsize*1.5)
            axs[1,2].set_ylabel("BOLD in $\Delta$%",fontsize=fontsize*1.25)
            axs[1,2].set_xlabel('time in s',fontsize=fontsize*1.25)
            axs[1,2].xaxis.set_tick_params(labelsize=fontsize)
            axs[1,2].yaxis.set_tick_params(labelsize=fontsize)
            fig.tight_layout()
            
            # plt.figure()
            # plt.plot(self.time_axis,bold*100)
            # plt.plot(self.time_axis,bold2*100)

            if save_plot:
                plt.figure(figsize=(6,3))
                plt.plot(self.time_axis, bold*100)
                plt.xlabel('time in s',fontsize=fontsize*1.25)
                plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
                plt.title('fMRI response',fontsize=fontsize*1.5)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                #plt.savefig('Model2_{}.png'.format(name))
                
        return 100*bold