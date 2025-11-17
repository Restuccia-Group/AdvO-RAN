from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 



from tf_agents import specs
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts



import config_file
import srslte_utils as sr 

import pandas as pd
import numpy as np
import pickle
import math


############################################################################################
FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST
autoencoder_filename = 'encoder.h5'

autoencoder = tf.keras.models.load_model(autoencoder_filename)

##########################################################################################

def compute_observation(pkl_filenames, n_slices, autoencoder):
    input_for_autoencoder = []
    data_tmp = []
    for i in range(n_slices):
        slice_based_data = read_pkl_and_select_sequential_rows(pkl_filenames[i],i,num_rows=10,add_prb_ratio=True)
        
        slice_based_data = np.array(slice_based_data[config_file.metric_list_autoencoder])
        data_tmp.append(slice_based_data)
        print(slice_based_data.shape)
        while slice_based_data.shape[0] < 10:
            slice_based_data = np.vstack((slice_based_data,np.zeros((1, slice_based_data.shape[1]))))

        input = scale_input_to_autoencoder(config_file.X_min,config_file.X_max, config_file.scale,slice_based_data)
        input_for_autoencoder.append(np.array(input))
    
    observation = [autoencoder.predict(np.expand_dims(input_for_autoencoder[i], axis=0)) for i in range(n_slices)]

    return np.array(observation).flatten().astype('float32'), data_tmp
    

def reward_function(metrics, n_slices):
    avg_metrics = np.array([np.mean(metrics[i]) for i in range(n_slices)])
    return avg_metrics, sum(config_file.weight_vector * avg_metrics)


def compute_reward(observation, remove_zero_entries, n_slices, reward_metrics):
    if remove_zero_entries:
        # we remove the entries that are all zeros and do not provide any info
        non_zero_entries = [
                observation[i][np.sum(observation[i],axis=1) > 0]
                for i in range(n_slices)]
        # add dummy entry for the case in which data for a specific slice has not been received
        for s_idx, s_el in enumerate(non_zero_entries):
            if s_el.shape[0] == 0:
                print('Adding dummy entries to compute reward slice ' + str(s_idx))
                non_zero_entries[s_idx] = np.zeros((1, len(reward_metrics)))
        metrics = [non_zero_entries[i][:][0][reward_metrics[i]] for i in range(n_slices)]
    else:
        metrics = [observation[i][:][reward_metrics[i]] for i in range(n_slices)]
    reward_per_slice, reward_avg = reward_function(metrics, n_slices)
    return reward_per_slice, reward_avg.astype('float32')


def scale_input_to_autoencoder(X_min=None,
                               X_max=None,
                               scale=None,
                               X=None):
    return [scale * (X[i] - np.array(X_min)) / (np.array(X_max) - np.array(X_min)) for i in range(len(X))]



def read_pkl_and_select_sequential_rows(file_path, slice_id, num_rows=10, add_prb_ratio=True):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    filtered_data = data[data['slice_id'] == slice_id]
    
    if len(filtered_data) < num_rows:
        raise ValueError(f"Not enough rows with slice_id {slice_id} to sample {num_rows} rows")
    
    indices = filtered_data.index.tolist()
    
    start_index = np.random.choice(indices[:-num_rows + 1])
    

    sequential_rows = filtered_data.loc[start_index:start_index + num_rows - 1]
    
   
    if add_prb_ratio:
        sequential_rows['ratio_granted_req'] = np.clip(np.nan_to_num(
            sequential_rows['sum_granted_prbs'] / sequential_rows['sum_requested_prbs']), a_min=0, a_max=1)
    
    return sequential_rows

def convert_prb_to_rgb(prb_allocation: list, du_prb: int=50) -> list:

    # size of RBG in PRBs (Table 7.6.1.1.1 - Type 0)
    rbg_size = 1
    if du_prb >= 11 and du_prb <= 26:
        rbg_size = 2
    elif du_prb >= 27 and du_prb <= 63:
        rbg_size = 3
    elif du_prb >= 64 and du_prb <= 110:
        rbg_size = 4
    else:
        logging.error('PRBs not supported ' + str(du_prb))
        return prb_allocation

    rbg_allocation = [math.ceil(x / rbg_size) for x in prb_allocation]
    return  rbg_allocation

def file_name_selector(slice_prb,scheduling_profile):
    
    pkl_filenames = []
    for i in range(len(config_file.slice_profiles.keys())):

        sched_folder = 'sched{}'.format(scheduling_profile[i])
        rgb_alloc = convert_prb_to_rgb(slice_prb,50)
        rgb_alloc = tuple(rgb_alloc)
        pkl_filename = '../agent_dataset/' + sched_folder +'/'+ config_file.rgb_tr_map_dict.get(rgb_alloc) + '.pkl'
        pkl_filenames.append(pkl_filename)

    return pkl_filenames


def reward_function(metrics, n_slices):
    avg_metrics = np.array([np.mean(metrics[i]) for i in range(n_slices)])
    return avg_metrics, sum(config_file.weight_vector * avg_metrics)


def compute_reward(observation, remove_zero_entries, n_slices, reward_metrics):
    if remove_zero_entries:
        # we remove the entries that are all zeros and do not provide any info
        non_zero_entries = [
                observation[i][np.sum(observation[i],axis=1) > 0]
                for i in range(n_slices)]
        # add dummy entry for the case in which data for a specific slice has not been received
        for s_idx, s_el in enumerate(non_zero_entries):
            if s_el.shape[0] == 0:
                print('Adding dummy entries to compute reward slice ' + str(s_idx))
                non_zero_entries[s_idx] = np.zeros((1, len(reward_metrics)))
        metrics = [non_zero_entries[i][:][0][reward_metrics[i]] for i in range(n_slices)]
    else:
        metrics = [observation[i][:][reward_metrics[i]] for i in range(n_slices)]
    reward_per_slice, reward_avg = reward_function(metrics, n_slices)
    return reward_per_slice, reward_avg.astype('float32')


def action_to_profile(action_id):
    idx_prb = int(np.floor(action_id/len(config_file.scheduling_combos)))
    idx_scheduling = int(action_id - len(config_file.scheduling_combos) * idx_prb)
    return config_file.feasible_prb_allocation[idx_prb], config_file.scheduling_combos[idx_scheduling]



class RanEnv(tf_environment.TFEnvironment):
    """
    Compute 
    
    
    """
    
    def __init__(self):

        observation_spec = specs.TensorSpec([],dtype=tf.float32,name='observations')
        action_spec = specs.TensorSpec([],dtype=tf.int32,name='actions')
        time_step_spec = ts.time_step_spec(observation_spec=observation_spec)
        slice_profiles = config_file.slice_profiles
        metric_dict = {"dl_buffer [bytes]": 1,
                    "tx_brate downlink [Mbps]": 2,
                    "tx_pkts downlink": 5,
                    "ratio_req_granted": 3,
                    "slice_id": 0,
                    "slice_prb": 4}
        metric_list = [config_file.slice_profiles[config_file.slice_names[config_file.slice_ids[i]]]['reward_metric'] for i in range(len(config_file.slice_names))]
        
        reward_metrics=[
            metric_dict[metric_list[i]]
            for i in range(len(config_file.slice_names))]
        self.reward_metrics_reduced_matrix = [0, 1, 2]

        

    def _current_time_step(self):
        """
        what is the last action
   
        """
        pass

    def _step(self, action):
        pass


    def _reset(self):
        start_prb_alloc = [6,39,5]
        start_sched = [0,0,1]
        start_pkl_filenames = file_name_selector(start_prb_alloc,start_sched)

        observation, data_tmp = compute_observation(start_pkl_filenames, n_slices=3,autoencoder=autoencoder)

        reward = compute_reward(observation=data_tmp,remove_zero_entries=False,n_slices=3,reward_metrics=self.reward_metrics_reduced_matrix)

        discount = tf.convert_to_tensor([0.99], dtype=tf.float32, name='discount')

        start_time_step = ts.TimeStep(FIRST,reward,discount,observation)

        return start_time_step
    




if __name__ == '__main__':


        env = RanEnv()

        start_time_step = env.reset()

        print(start_time_step.observation)



