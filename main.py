# -*- coding: utf-8 -*-
"""
Author: Lea Demelius
"""
from LinearTransformModel import LinearTransformModel
from BalloonModel import BalloonModel
from MechanisticModel import MechanisticModel
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os
import pandas as pd

def dIdt(I, t, params):
    stim_func, tau1, kappa = params
    return 1/tau1*(kappa*stim_func(t)-I*(kappa+1))
    
def create_stimulus(time, onset, offset):
    stimulus = np.array([0.]*len(time))
    stimulus[(time <= offset) & (time >= onset)] = 1
    return stimulus

def create_ONOFF_stimulus(time, onset, stim_duration, inter_duration, reps, a=1):
    stimulus = np.array([0.]*len(time))
    #stimulus = np.random.uniform(0.,0.1,len(time))
    for i in range(reps):
        stimulus[(time >= onset + i*(stim_duration+inter_duration)) & \
                 (time <= (onset+stim_duration+i*(stim_duration+inter_duration)))] = a
    return stimulus

def average_spikes(z, time, t_bin=100):
    temp_avg = np.count_nonzero(z[:t_bin,:],axis=0) 
    for i in range(1,int(time/t_bin)):
        temp_avg = np.vstack((temp_avg, np.count_nonzero(z[i*t_bin:(i+1)*t_bin,:],axis=0)))
    temp_avg = temp_avg*1000/t_bin #to get spikes/s
    firing_rate = np.mean(temp_avg, axis=1) #population rate
    return firing_rate

def plot_stimulus(time_axis, fr_stimulus, name):
    fontsize=12.5
    plt.figure(figsize=(4.5,3))
    plt.plot(time_axis, fr_stimulus)
    plt.xlabel('time in s',fontsize=fontsize*1.25)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel('firing rate in spikes/s',fontsize=fontsize*1.25)
    plt.title('simulated firing rate',fontsize=fontsize*1.5)
    if save_plot:
        plt.savefig('{}_firingrate.png'.format(name))

def simulate_models(path, split_mode, t_bin, fr_input):

    if fr_input:
        if split_mode == 'all':
            loc = 'down'
            fr_stimulus = np.load(os.path.join(path, 'fr_all_{}.npy'.format(loc)))
            if t_bin == 2000:
                fr_stimulus = [(a + b) / 2 for a, b in zip(fr_stimulus[::2], fr_stimulus[1::2])]
            fraction_stimulus = fr_stimulus/max(fr_stimulus)
            time = len(fr_stimulus)*t_bin
            name = 'total'
            
            time_axis = np.arange(0, time, t_bin)/1000
            
            if plot:
                plot_stimulus(time_axis, fr_stimulus, name)
                
            model1 = LinearTransformModel(time_axis, fr_stimulus)
            bold1 = model1.solve(plot, save_plot, name)
            
            model2 = BalloonModel(time_axis, fr_stimulus, outflow_linear=False)
            bold2 = model2.solve(plot, save_plot, name)
            
            model3 = MechanisticModel(time_axis, fraction_stimulus, fraction_stimulus)
            bold3 = model3.solve(plot, save_plot, name) 
            
            # path = '/calc/demelius/experimental_data/final_figures/figures_results'
            # np.save(os.path.join(path, '8bars_{}/LTM.npy'.format(loc)), bold1)
            # np.save(os.path.join(path, '8bars_{}/Balloon.npy'.format(loc)), bold2)
            # np.save(os.path.join(path, '8bars_{}/Mechanistic.npy'.format(loc)), bold3)
            
            fontsize = 10
            plt.figure(figsize=(12,3))
            plt.plot(time_axis, bold1, 'purple')
            plt.plot(time_axis, bold2,'orange')
            plt.plot(time_axis, bold3,'green')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('time in s',fontsize=fontsize*1.25)
            plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
            plt.title('simulated fMRI responses',fontsize=fontsize*1.5)
            plt.legend(['Linear transform model','Balloon model','Mechanistic model'],fontsize=fontsize*1.25,loc='lower right',bbox_to_anchor=(1, 1)) 
            #plt.legend(['Linear transform model','Balloon model','Mechanistic model'],fontsize=fontsize*1.25,loc='lower center',bbox_to_anchor=(0.5, 1.2),ncol=3)
        
        elif split_mode == 'laminar_third':
            layer_names = ['superficial layer', 'middle layer','deep layer']
              
            bold1s = []
            bold2s = []
            bold3s = []
            loc = 'up'
            if t_bin == 2000:
                t_bin1 = 1000
            fr_upper = np.load(os.path.join(path, 'fr_laminar_{}_tbin{}_upper.npy'.format(loc,t_bin1)))
            fr_middle = np.load(os.path.join(path, 'fr_laminar_{}_tbin{}_middle.npy'.format(loc,t_bin1)))
            fr_lower = np.load(os.path.join(path, 'fr_laminar_{}_tbin{}_lower.npy'.format(loc,t_bin1)))
            if t_bin == 2000:
                fr_upper = [(a + b) / 2 for a, b in zip(fr_upper[::2], fr_upper[1::2])]
                fr_middle = [(a + b) / 2 for a, b in zip(fr_middle[::2], fr_middle[1::2])]
                fr_lower = [(a + b) / 2 for a, b in zip(fr_lower[::2], fr_lower[1::2])]
            max_val = np.max([np.max(fr_upper),np.max(fr_middle),np.max(fr_lower)])
            for fr_stimulus, name in zip([fr_upper, fr_middle, fr_lower],layer_names):
                time = len(fr_stimulus)*t_bin
                fraction_stimulus = fr_stimulus/max_val
                print(max(fr_stimulus))
                
                time_axis = np.arange(0, time, t_bin)/1000
                
                if plot:
                    plot_stimulus(time_axis, fr_stimulus, name)
                    
                model1 = LinearTransformModel(time_axis, fr_stimulus) 
                bold1 = model1.solve(plot, save_plot, name)
                bold1s.append(bold1)
                
                model2 = BalloonModel(time_axis, fr_stimulus, outflow_linear=False)
                bold2 = model2.solve(plot, save_plot, name)
                bold2s.append(bold2)
                
                
                model3 = MechanisticModel(time_axis, fraction_stimulus, fraction_stimulus)
                bold3 = model3.solve(plot, save_plot, name) 
                bold3s.append(bold3)
          
            fontsize = 10
            
            plt.figure(figsize=(12,2))
            for fr, color in zip([fr_upper, fr_middle, fr_lower],['blue','red','gold']):
                plt.plot(time_axis, fr,color)
            plt.xlabel('time in s',fontsize=fontsize*1.25)
            plt.ylabel('firing rates in spikes/s',fontsize=fontsize*1.25)
            plt.legend(layer_names,fontsize=fontsize*1.25,loc='lower center',bbox_to_anchor=(0.5, 1.2),ncol=3)
            #plt.legend(layer_names,fontsize=fontsize*1.25,loc='lower center',bbox_to_anchor=(0.5, 1.2),ncol=3)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('simulated firing rate',fontsize=fontsize*1.5)
            
            plt.figure(figsize=(12,2))
            for bold1, color in zip(bold1s,['blue','red','gold']):
                plt.plot(time_axis, bold1,color)
            #plt.legend(layer_names,fontsize=fontsize*1.25,loc='upper right')
            plt.xlabel('time in s',fontsize=fontsize*1.25)
            plt.title('laminar fMRI responses: Linear transform model',fontsize=fontsize*1.5)
            plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            
            plt.figure(figsize=(12,2))
            for bold2, color in zip(bold2s,['blue','red','gold']):
                plt.plot(time_axis, bold2,color)
            #plt.legend(layer_names,fontsize=fontsize*1.25,loc='upper right')
            plt.xlabel('time in s',fontsize=fontsize*1.25)
            plt.title('laminar fMRI responses: Balloon model',fontsize=fontsize*1.5)
            plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            
            plt.figure(figsize=(12,2))
            for bold3, color in zip(bold3s,['blue','red','gold']):
                plt.plot(time_axis, bold3, color)
            #plt.legend(layer_names,fontsize=fontsize*1.25,loc='upper right')
            plt.xlabel('time in s',fontsize=fontsize*1.25)
            plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
            plt.title('laminar fMRI responses: Mechanistic model',fontsize=fontsize*1.5)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            
        else:
            raise Exception('Option not available.')
        
        
        
            
    else:
        spikes = np.load(os.path.join(path, 'spike_trains_all.npy')) 
        print(np.shape(spikes))
        time = np.shape(spikes)[0] #in ms
         
        
        x_coords = np.load(os.path.join(path, 'x_coords.npy'))
        z_coords = np.load(os.path.join(path, 'z_coords.npy'))
        
        sel1 = True
        if sel1: #manual neuron selection on cortical plane according to azimuth/elevation relationship)
            xz_mask = np.logical_and(np.logical_and(x_coords>=-42, x_coords<=42),np.logical_and(z_coords>-75, z_coords<=+75))
        else: #connectivity based selection
            import pickle as pkl
            import h5py
            data_path = '/home/demelius/tf_billeh_column/billeh_data/GLIF_network'
            with open(os.path.join(data_path,'network_dat.pkl'),'rb') as f:
                network_d = pkl.load(f)
            network_h5 = h5py.File(os.path.join(data_path, 'network/v1_nodes.h5'), mode='r')
            with open(os.path.join(data_path,'input_dat.pkl'),'rb') as f:
                input_d = pkl.load(f)
            input_h5 = h5py.File(os.path.join(data_path,'network/lgn_nodes.h5'), mode='r')
            
            x, y, z = [np.array(network_h5['nodes']['v1']['0'][k]) for k in 'xyz']
            # x_new = x[np.sqrt(x**2 + z**2) <= 400]
            # z_new = z[np.sqrt(x**2 + z**2) <= 400]
            # print(np.array_equal(x_coords,x_new))
            # print(np.array_equal(z_coords,z_new))
            # import pdb; pdb.set_trace()
            
            lgn_x, lgn_y = [np.array(input_h5['nodes']['lgn']['0'][k]) for k in 'xy']
            
            post_indices = []
            pre_indices = []
            weights = []
            for edge in input_d[0][1]:
                target_tf_id = np.array(edge['target'])
                source_tf_id = np.array(edge['source'])
                weights_tf = np.array(edge['params']['weight'])
                post_indices.extend(target_tf_id)
                pre_indices.extend(source_tf_id)
                weights.extend(weights_tf)
            input_indices = np.stack([post_indices, pre_indices], -1)
            input_weights = np.array(weights)
            
            # def get_pre_lgn_node_ids(v1_mask):
            #     sel_node_ids = np.where(v1_mask)[0]
            #     input_weight_mask = np.zeros_like(input_weights, dtype=bool)
            #     for sel_node_id in sel_node_ids:
            #         input_weight_mask = np.logical_or(input_weight_mask, input_indices[:,0] == sel_node_id)
            #     pre_node_ids = input_indices[input_weight_mask, 1]
            #     return np.unique(pre_node_ids)
            
            # core_mask = np.sqrt(x**2 + z**2) <= 400
            # lgn_node_ids = get_pre_lgn_node_ids(core_mask)
            # lgn_x_core = lgn_x[lgn_node_ids]
            # lgn_y_core = lgn_y[lgn_node_ids]
            
            lgn_sel = np.logical_and(np.logical_and(lgn_x>=120-27,lgn_x<=120+27),np.logical_and(lgn_y>=-3,lgn_y<60+3))
            
            def get_v1_node_ids(lgn_sel):
                sel_node_ids = np.where(lgn_sel)[0]
                input_weight_mask = np.zeros_like(input_weights, dtype=bool)
                for sel_node_id in sel_node_ids:
                    input_weight_mask = np.logical_or(input_weight_mask, input_indices[:,1] == sel_node_id)
                v1_node_ids = input_indices[input_weight_mask, 0]
                return np.unique(v1_node_ids)
            
            v1_node_ids = get_v1_node_ids(lgn_sel)
            xz_mask = np.zeros_like(x,bool)
            xz_mask[v1_node_ids] = True
            xz_mask = xz_mask[np.sqrt(x**2 + z**2) <= 400]
            
        e_mask = np.load(os.path.join(path, 'e_mask.npy'))
    
        if split_mode == 'all':
            name = 'total'
            spikes_total = spikes[:,xz_mask]
            N = np.shape(spikes_total)[1] 
            print('Number of neurons: {}'.format(N))
            spikes_exc = spikes[:,np.logical_and(xz_mask,e_mask)]
            spikes_inh = spikes[:,np.logical_and(xz_mask,~e_mask)]
            N_exc = np.shape(spikes_exc)[1] 
            N_inh = np.shape(spikes_inh)[1] 
            print('Number of excitatory neurons: {}'.format(N_exc))
            print('Number of inhibitory neurons: {}'.format(N_inh))
            
            fr_stimulus = average_spikes(spikes_total, time, t_bin)
            fraction_stimulus = fr_stimulus/max(fr_stimulus) #/2 #/N*100
            
            fr_stimulus_exc = average_spikes(spikes_exc, time, t_bin)
            fraction_stimulus_exc = fr_stimulus_exc/max(fr_stimulus) #2 #N_exc*100
            
            fr_stimulus_inh = average_spikes(spikes_inh, time, t_bin)
            fraction_stimulus_inh = fr_stimulus_inh/max(fr_stimulus) #2 #N_inh*100
        
            time_axis = np.arange(0, time, t_bin)/1000
            
            if plot:
                plot_stimulus(time_axis, fr_stimulus, name)
            
            #path = '/calc/demelius/experimental_data/6.4.22'
            #np.save(os.path.join(path, 'fr_pred_1bar_down.npy'), fr_stimulus)
                
            model1 = LinearTransformModel(time_axis, fr_stimulus)
            bold1 = model1.solve(plot, save_plot, name)
            
            #path = '/calc/demelius/experimental_data/6.4.22'
            #np.save(os.path.join(path, 'fmri_pred_LTM_1bar_down.npy'), fr_stimulus)
            
            model2 = BalloonModel(time_axis, fr_stimulus, outflow_linear=False)
            bold2 = model2.solve(plot, save_plot, name)
            
            #path = '/calc/demelius/experimental_data/6.4.22'
            #np.save(os.path.join(path, 'fmri_pred_Balloon_1bar_down.npy'), fr_stimulus)

            
            model3 = MechanisticModel(time_axis, fraction_stimulus, fraction_stimulus)
            bold3 = model3.solve(plot, save_plot, name) 
            
            #path = '/calc/demelius/experimental_data/6.4.22'
            #np.save(os.path.join(path, 'fmri_pred_Mech_1bar_down.npy'), fr_stimulus)
            
            fontsize=11.25
            plt.figure(figsize=(6,3))
            plt.plot(time_axis, bold1, 'purple')
            plt.plot(time_axis, bold2,'orange')
            plt.plot(time_axis, bold3,'green')
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlabel('time in s',fontsize=fontsize*1.25)
            plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
            plt.title('simulated fMRI responses',fontsize=fontsize*1.5)
            plt.legend(['Linear transform','Balloon','Mechanistic'],fontsize=12.5) #,'3','3.2'])
                
        else:
            if split_mode == 'laminar_layer':
                supra_mask = np.load(os.path.join(path, 'supra_mask.npy'))
                granular_mask = np.load(os.path.join(path, 'granular_mask.npy'))
                infra_mask = np.load(os.path.join(path, 'infra_mask.npy'))
                layer_masks = [supra_mask, granular_mask, infra_mask]
                layer_names = ['supragranular', 'granular','infragranular']
            elif split_mode == 'laminar_third':
                depth_coords = np.load(os.path.join(path, 'y_coords.npy'))
                lower_mask = np.logical_and(depth_coords < -566.6, depth_coords >=- 850)
                middle_mask = np.logical_and(depth_coords < -283.3, depth_coords >= -566.6)
                upper_mask = np.logical_and(depth_coords < 0, depth_coords >= -283.3)  
                layer_masks = [upper_mask, middle_mask, lower_mask]
                layer_names = ['upper', 'middle','lower']
            else:
                raise Exception('Option not available.')
    
    
            bold1s = []
            bold2s = []
            bold3s = []
            firing_rates = []
            for layer_mask, name in zip(layer_masks,layer_names):
                coord_mask = np.logical_and(xz_mask, layer_mask)
                spikes_total = spikes[:,coord_mask]
                N = np.shape(spikes_total)[1] 
                print('Number of neurons: {}'.format(N))
                spikes_exc = spikes[:,np.logical_and(coord_mask,e_mask)]
                spikes_inh = spikes[:,np.logical_and(coord_mask,~e_mask)]
                N_exc = np.shape(spikes_exc)[1] 
                N_inh = np.shape(spikes_inh)[1] 
                print('Number of excitatory neurons: {}'.format(N_exc))
                print('Number of inhibitory neurons: {}'.format(N_inh))
                
                fr_stimulus = average_spikes(spikes_total, time, t_bin)
                fraction_stimulus = fr_stimulus/1.626 #max(fr_stimulus) #N*100
                print(max(fr_stimulus))
                firing_rates.append(fr_stimulus)
                #np.save('/home/demelius/experimental_data/30.3.22/firingrate_tbin10_{}'.format(name), fr_stimulus)
                
                fr_stimulus_exc = average_spikes(spikes_exc, time, t_bin)
                fraction_stimulus_exc = fr_stimulus_exc/3 #max(fr_stimulus) #2 #N_exc*100
                
                fr_stimulus_inh = average_spikes(spikes_inh, time, t_bin)
                fraction_stimulus_inh = fr_stimulus_inh/3 #max(fr_stimulus) #2 #N_inh*100
            
                time_axis = np.arange(0, time, t_bin)/1000
                
                if plot:
                    plot_stimulus(time_axis, fr_stimulus, name)
                    
                model1 = LinearTransformModel(time_axis, fr_stimulus)
                bold1 = model1.solve(plot, save_plot, name)
                bold1s.append(bold1)
                
                model2 = BalloonModel(time_axis, fr_stimulus, outflow_linear=False)
                bold2 = model2.solve(plot, save_plot, name)
                bold2s.append(bold2)
                
                model3 = MechanisticModel(time_axis, fraction_stimulus, fraction_stimulus)
                bold3 = model3.solve(plot, save_plot, name) 
                bold3s.append(bold3)
          
            plt.figure(figsize=(6,4))
            for fr in firing_rates:
                plt.plot(time_axis, fr)
            plt.xlabel('time in s')
            plt.ylabel('Laminar firing rates in spikes/s')
            
            plt.legend(layer_names)
            
            plt.figure(figsize=(8,6))
            for bold1 in bold1s:
                plt.plot(time_axis, bold1)
            plt.legend(layer_names)
            plt.xlabel('time in s')
            plt.title('Linear transform model')
            plt.ylabel('Laminar bold signals in %')
            
            plt.figure(figsize=(8,6))
            for bold2 in bold2s:
                plt.plot(time_axis, bold2)
            plt.legend(layer_names)
            plt.xlabel('time in s')
            plt.title('Balloon model')
            plt.ylabel('Laminar bold signals in %')
            
            plt.figure(figsize=(8,6))
            for bold3 in bold3s:
                plt.plot(time_axis, bold3)
            plt.legend(layer_names)
            plt.xlabel('time in s')
            plt.ylabel('Laminar bold signals in %')
            plt.title('Mechanistic model')
        
# -------------------------------------------------------------------------------------------------
    

plt.close('all')

artificial_stimulus = False

plot=True
save_plot=False

if artificial_stimulus:
    which = 'artificial'
    sim_duration = 20
    time_axis = np.arange(0, sim_duration, 0.001) #0.001) #0.001
    onset = 1
    stim_duration = 0.5
    inter_duration = 1
    reps = 1
    amplitude = 20
    assert (onset + reps*(stim_duration+inter_duration)) <= sim_duration
    stimulus = create_ONOFF_stimulus(time_axis, onset, stim_duration, inter_duration, reps, a=amplitude)
    
    adaption = False
    if adaption:
        stim_func = interp1d(time_axis,stimulus,bounds_error=False, fill_value="extrapolate")
        params = [stim_func,3.,3.]
        I = np.squeeze(odeint(dIdt, 0., time_axis, args=(params,)))
        #no negative neural response:
        mask = np.where(stimulus > 0)
        new_stimulus = stimulus[mask] - I[mask]
        stimulus[mask] = new_stimulus
        
    fontsize = 11.25
    
    if plot:
        plt.figure(figsize=(6,3))
        plt.plot(time_axis, stimulus/amplitude)
        plt.xlabel('time in s',fontsize=fontsize*1.25)
        plt.ylabel('amplitude',fontsize=fontsize*1.25)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('artificial stimulus',fontsize=fontsize*1.5)
        #plt.ylabel('{} stimulus'.format(which))
        if save_plot:
            plt.savefig('Artificial_Stimulus.png')
    
    model1 = LinearTransformModel(time_axis, stimulus)
    bold1 = model1.solve(plot, save_plot)
    
    model2 = BalloonModel(time_axis, stimulus, outflow_linear=False)
    bold2 = model2.solve(plot, save_plot)
    
    model3 = MechanisticModel(time_axis, stimulus/amplitude, stimulus/amplitude)
    bold3 = model3.solve(plot, save_plot) 
    
    plt.figure(figsize=(9,3))
    plt.plot(time_axis, bold1,'purple')
    plt.plot(time_axis, bold2,'orange')
    plt.plot(time_axis, bold3,'green')
    plt.xlabel('time in s',fontsize=fontsize*1.25)
    plt.ylabel('BOLD in $\Delta$%',fontsize=fontsize*1.25)
    plt.title('fMRI responses',fontsize=fontsize*1.5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(['Linear transform model','Balloon model','Mechanistic model'],fontsize=fontsize*1.25,loc='upper right')

## Billeh stimulus   
else:
    fr_input = True
    split_mode = 'all' #'all', 'laminar_layer', 'laminar_third'
    t_bin = 2000 #in ms
    path = '/home/demelius/spiketrains_for_fMRI/MuckliStim_532s_0bis1_spikes'
    simulate_models(path, split_mode, t_bin, fr_input)

