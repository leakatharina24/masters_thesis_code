"""
Credit to: Sten et al., “Neural inhibition can explain negative bold responses: A mechanistic modelling
and fmri study,” Neuroimage, vol. 158, pp. 219–231, 2017.

Adapted by: Lea Demelius
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class MechanisticModel:
    def __init__(self, time_axis, img_stimulus, exc, inh = None):
        self.time_axis = time_axis
        self.stimulus = img_stimulus
        self.excStim = exc
        self.inhStim = inh
        
    def interpolate(self, stimulus):
        return interp1d(self.time_axis,stimulus,bounds_error=False, fill_value="extrapolate", kind='nearest')
    
    def f(self,y,t,params): 
        #states 
        Glutamate, GABA, Delay_m, Ca2, AA, AAmetvc1, AAmetvc2, AAmetvc3, AAmetvc4, AAmetvd1, AAmetvd2, AAmetvd3, AAmetvd4, oHb, dHb, O2A, O2B, GlucoseA, GlucoseB = y
        #stimulus
        Stim_img = self.interpolate(self.stimulus)(t) #not perfect, esp. for rectangular stimulus
        Stim_glut = self.interpolate(self.excStim)(t)
        try:
            Stim_gaba = self.interpolate(self.inhStim)(t)
        except:
            Stim_gaba = Stim_glut
    
        #params 
        k_met, k_Glutamate, k_GABA, sink_met, sink_Glutamate, sink_GABA, sink_Ca, k_ca, k3, k4, k_PL, k_vc, k_vd, k_vc1, k_vc2, k_vc3, k_vc4, k_vd1, k_vd2, k_vd3, k_vd4, k_flow, k_bv, k_s, k_i, k1f, k1b, k_O2, k_gluc1, k_gluc2, k_m, k_basalmet, prop1, prop2 = params 
    
        #variables 
        GlucoseB_basal, oHb_basal, O2B_basal, dHb_basal = [10, 10, 10, 10]
    
        v_Glut = k3*Glutamate    
        v_GABA = GABA/k4
        V_flow = k_flow + np.exp(k_s*AAmetvd4 - k_i*AAmetvc4)
        V_bv = np.power(V_flow, k_bv)
            
        #reactions 
        input1 = k_met*Stim_img
        input2 = k_Glutamate*Stim_glut
        input3 = k_GABA*Stim_gaba
    
        Ca_in = k_ca*(1 + v_Glut)*(1/(1 + v_GABA*1)) 
    
        v_vc1 = k_vc1*AAmetvc1
        v_vc2 = k_vc2*AAmetvc2
        v_vc3 = k_vc3*AAmetvc3
        v_vc4 = k_vc4*AAmetvc4
        v_vd1 = k_vd1*AAmetvd1
        v_vd2 = k_vd2*AAmetvd2
        v_vd3 = k_vd3*AAmetvd3
        v_vd4 = k_vd4*AAmetvd4
    
        v1f = k1f*oHb
        v1b = k1b*dHb*O2B
        vinoHb = V_flow*oHb_basal
        voutoHb = V_flow*oHb
        vindHb = V_flow*dHb_basal
        voutdHb = V_flow*dHb
        vinO2 = V_flow*O2B_basal
        voutO2 = V_flow*O2B
        vO2A = k_O2*(O2B - O2A)
    
        vinG = V_flow*GlucoseB_basal
        voutG = V_flow*GlucoseB
        vGluc1 = k_gluc1*(GlucoseB - GlucoseA)
        vGluc2 = k_gluc2*(GlucoseB/(k_m + GlucoseB))
    
        vbasalmet = k_basalmet*GlucoseA*np.power(O2A, prop1)
        vstimmet = Delay_m*GlucoseA*np.power(O2A, prop2)
    
        dt_glut = input2 - sink_Glutamate*Glutamate
        dt_gaba = input3 - sink_GABA*GABA
        dt_met = input1 - sink_met*Delay_m
        dt_ca = Ca_in - sink_Ca*Ca2
        dt_AA = k_PL*Ca2 - AA*(k_vc + k_vd)
        dt_AAc1 = AA*k_vc - v_vc1
        dt_AAc2 =  v_vc1 - v_vc2
        dt_AAc3 = v_vc2 - v_vc3
        dt_AAc4 = v_vc3 - v_vc4
        dt_AAd1 = AA*k_vd - v_vd1
        dt_AAd2 = v_vd1 - v_vd2
        dt_AAd3 = v_vd2 - v_vd3
        dt_AAd4 = v_vd3 - v_vd4
        dt_oHb = v1b - v1f + vinoHb - voutoHb
        dt_dHb = v1f - v1b + vindHb - voutdHb
        dt_O2A = V_bv*vO2A - vbasalmet*prop1 - vstimmet*prop2
        dt_O2B = v1f - v1b - vO2A + vinO2 - voutO2
        dt_GlucA = (vGluc1 + vGluc2)*V_bv - vbasalmet - vstimmet
        dt_GlucB = -(vGluc1 + vGluc2) + vinG - voutG
    

        dy = [
            dt_glut, dt_gaba, dt_met, dt_ca, dt_AA,dt_AAc1,
            dt_AAc2, dt_AAc3, dt_AAc4, dt_AAd1, dt_AAd2, dt_AAd3, dt_AAd4, 
            dt_oHb, dt_dHb, dt_O2A, dt_O2B, dt_GlucA, dt_GlucB
            ]
        return dy

    def solve(self, plot=True, save_plot=False, name=''):
        names = ['Glutamate', 'GABA', 'Delay_m', 'Ca2', 'AA', 'AAmetvc1', 'AAmetvc2', 'AAmetvc3', 'AAmetvc4', 'AAmetvd1', 'AAmetvd2', 'AAmetvd3', 'AAmetvd4', 'oHb', 'dHb', 'O2A', 'O2B', 'GlucoseA', 'GlucoseB']
        y0 = [0, 0, 0, 3.23, 6215.57, 3212.53, 7648.15, 3487.25, 6373.3, 5217.49, 5105.26, 4798.85, 2321.25, 4.5397, 15.4603, 1.3336,10.6676, 9.3039, 9.3843]
        
        params = [2919.605241, 23.709923, 3.93136, 1.016485, 65.82106, 50, 1.545044, 4.987185, 4, 1, 2172.361112, 0.608611, 0.519536, 1.177533, 0.494612, 1.084768, 0.593544, 0.61892, 0.632527, 0.672914, 1.39115, 8522.963175, 0.179979, 0.139901, 0.050014, 44011.61482,  916.074176, 4581.212879, 68295.66808, 3097.905391, 68397.64798, 322.748711, 7.784567, 6.393969]

        solver = odeint(self.f, y0, self.time_axis, args=(params,), hmax=0.05)

        k_y = 0.210053
        BOLD_0 = np.exp(-k_y*y0[-5])  
        BOLD = np.exp(-k_y*solver[:,-5])
        
        if plot:
            fontsize = 11.25
            fig, axs = plt.subplots(4,5,figsize=(12,12))
            x = 0
            for i in range(len(names)):
                y = i%5
                axs[x,y].plot(self.time_axis,solver[:,i])
                #axs[x,y].legend([names[i]],fontsize=fontsize*1.25) 
                axs[x,y].set_title(names[i],fontsize=fontsize*1.5)
                axs[x,y].xaxis.set_tick_params(labelsize=fontsize)
                axs[x,y].yaxis.set_tick_params(labelsize=fontsize)
                axs[x,y].set_xlabel('time in s',fontsize=fontsize*1.25)
                if y ==4:
                    x +=1
            axs[3,4].plot(self.time_axis, ((BOLD - BOLD_0)/BOLD_0)) #*100))
            #axs[3,4].legend(['fMRI response'],fontsize=fontsize*1.25)
            axs[3,4].set_title('fMRI response',fontsize=fontsize*1.5)
            axs[3,4].set_ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
            axs[3,4].xaxis.set_tick_params(labelsize=fontsize)
            axs[3,4].yaxis.set_tick_params(labelsize=fontsize)
            plt.tight_layout()
    
            if save_plot:
                plt.figure(figsize=(6,3))
                plt.plot(self.time_axis, ((BOLD - BOLD_0)/BOLD_0)) #*100))
                plt.xlabel('time in s',fontsize=fontsize*1.25)
                plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
                plt.title('fMRI response',fontsize=fontsize*1.5)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                #plt.savefig('Model3_{}.png'.format(name))
                
        return (BOLD-BOLD_0)/BOLD_0 #*100