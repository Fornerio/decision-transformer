import os
import sys
import argparse
import gym
import tqdm
import pprint
import highway_env
from gym.wrappers import RecordVideo
import numpy as np
import torch
from decision_transformer.models.decision_transformer import DecisionTransformer
import pickle
# Evaluation
from highway_env.envs.common.evaluate import PrintMetrics
import random

ACTIONS_ALL = {0: 'LANE_LEFT',1: 'IDLE',2: 'LANE_RIGHT',3: 'FASTER',4: 'SLOWER'}
metricObj = PrintMetrics()

def make_highway_env(env_name):
    env = gym.make(env_name)
    env.unwrapped.configure({
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 7,
            'features': [
                'presence',
                'x',
                'y',
                'vx',
                'vy'
            ],
            'absolute': False
        },
        'duration': 60,
        'simulation_frequency': 15})

    print(env.observation_space)
    return env

# load training sequences
def load_sequence(row, collisions=True):
    '''
    Load a sequence from a row of the dataset.
    :param row: [[state1, action1, reward1], [state2, action2, reward2], ..., [stateN, actionN, rewardN]]
    :return: sequence: dict containing {'states': np.array([state1, state2, ..., stateT]),
                                       'actions': np.array([action1, action2, ..., actionT]),
                                       'rewards': np.array([reward1, reward2, ..., rewardT]),
                                       'dones': np.array([0,0, ..., 1])} -> trivial for our case as we always have one
                                       scene for each episode. Dones is also not used in experiments.
                    states: np.array of shape (T, *state_dim)
                    actions: np.array of shape (T, *action_dim)
                    rewards: np.array of shape (T, )
                    dones: np.array of shape (T, )
    '''
    states = []
    actions = []
    rewards = []
    for state, action, reward in row[:-1]:
        # flatten state for mlp encoder
        states.append(state.reshape(-1))
        one_hot_action = np.zeros(3)
        one_hot_action[action] = 1
        actions.append(one_hot_action)
        rewards.append(reward)

    states = np.array(states)
    actions = np.array(actions)

    rewards = np.array(rewards)

    dones = np.zeros_like(rewards)
    dones[-1] = 1
    sequence = {'states': states, 'actions': actions, 'rewards': rewards, 'dones': dones}
    return sequence

def evaluate_model(variant):

    if variant['env'] == 'dm_env':
        env = make_highway_env('dm-env-v0')
        env.reset()
    else:
        raise NotImplementedError
        
    # Load sequences
    A = np.load(variant['dataset'], allow_pickle=True)
    sequences = [load_sequence(row) for row in A] #if row[-1] is False]

    # get some useful statistics from training set
    max_ep_len = max([len(path['states']) for path in sequences])  # take it as the longest trajectory
    print(max_ep_len)

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)

    # load model
    checkpoint = torch.load(variant['model'], map_location=variant['device'], pickle_module=pickle)
    render = variant['render']


    state, done = env.reset(), False
    act_dim = np.squeeze(sequences[0]['actions'].shape[1:])
    state_dim = np.squeeze(sequences[0]['states'].shape[1:])

    model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )

    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    scale = np.mean([len(path['states']) for path in sequences])  # scale for rtg

    # save all sequence information into separate lists
    states, traj_lens, returns = [], [], []
    for path in sequences:
        if variant['mode'] == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['states'])
        traj_lens.append(len(path['states']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    max_return = np.max(returns)

    print('max_return: ',max_return)
    print('mean_return: ',np.mean(returns))
    print('min_return: ',np.min(returns))

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)
    print('num_timesteps: ',num_timesteps)

    model.eval()
    model.to(device=variant['device'])

    state_mean = torch.from_numpy(state_mean).to(device=variant['device'])
    state_std = torch.from_numpy(state_std).to(device=variant['device'])

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=variant['device'], dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=variant['device'], dtype=torch.float32)
    rewards = torch.zeros(0, device=variant['device'], dtype=torch.float32)

    target_return = max_return

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=variant['device'], dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=variant['device'], dtype=torch.long).reshape(1, 1)

    sim_states = []

    # Initialize totals
    total_ep_return = 0

    for episodes in tqdm.tqdm(range(variant['num_episodes'])):

        if variant['env'] == 'dm_env':
            env = gym.make('dm-env-v0')
            env.reset()
        else:
            raise NotImplementedError

        env.config['duration']=58
            
        state = env.reset()
        done = False

        episode_return, episode_length = 0, 0
        total_rewards = []
        total_reward = 0

        decision_change_num, left_lane_change_num, right_lane_change_num, episode_reward = 0, 0, 0, 0
        accelerations, decelerations, speeds = [], [], []
        last_lane_idx = None
        last_action = ""

        t=0

        while not done:
            if variant['render']:
                env.render()
            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=variant['device'])], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=variant['device'])])

            # num_embeddings = model.time_embeddings.num_embeddings  # Adjust this to match your model's attribute
            # max_valid_index = num_embeddings - 1

            # # Ensure timesteps are within the valid range
            # timesteps = torch.clamp(timesteps, 0, max_valid_index)

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            optimal_action = np.argmax(action)
            
            state, reward, done, info  = env.step(optimal_action)

            if last_action != env.vehicle.current_action and last_action != "":
                decision_change_num += 1
            last_action = env.vehicle.current_action    
            
            speeds.append(env.vehicle.speed)

            if env.vehicle.throttle > 0:
                accelerations.append(env.vehicle.throttle)
            else:
                decelerations.append(env.vehicle.throttle)

            if last_lane_idx != env.vehicle.lane_index[2] and last_lane_idx is not None:
                if last_lane_idx > env.vehicle.lane_index[2]:
                    left_lane_change_num += 1
                else:
                    right_lane_change_num += 1
            last_lane_idx = env.vehicle.lane_index[2]

            cur_state = torch.from_numpy(state).to(device=variant['device']).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1]

            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)

            timesteps = torch.cat(
                [timesteps,
                torch.ones((1, 1), device=variant['device'], dtype=torch.long) * (t + 1)], dim=1)
            
            #print(timesteps)

            episode_return += reward
            t+=1

        episode_duration = (env.steps/env.config["policy_frequency"])
        left_lane_change_rate = left_lane_change_num / episode_duration
        right_lane_change_rate = right_lane_change_num / episode_duration
        mean_speed = np.mean(speeds)
        km_travelled = (mean_speed * episode_duration)/1000
        mean_acceleration = np.mean(accelerations)
        mean_deceleration = np.mean(decelerations)
        metricObj.saveEpisodeData(env_name='dm-env-v0', km_travelled=km_travelled, decision_change_num=decision_change_num, left_lane_change_num=left_lane_change_num,\
            right_lane_change_num=right_lane_change_num, mean_speed=mean_speed*3.6, mean_acceleration=mean_acceleration, mean_deceleration=mean_deceleration,\
            collision=env.vehicle.crashed, episode_duration=episode_duration, curr_episode_num=episodes+1)
        
        # Update episode metrics
    #     ep_returns.append(episode_return)
    #     ep_len.append(episode_length)
    #     #mean_ep_speeds = np.mean(total_speed)
    #     crash_counter += int(crashed)

    #     # Update totals
    #     total_ep_length += episode_length
    #     total_ep_return += episode_return
    #     #total_speeds.append(mean_ep_speeds)

    # # Calculate averages and rates
    # mean_ep_length = total_ep_length / num_episodes
    # #mean_speed = np.mean(total_speeds) # Total speed divided by total lengths of episodes
    # collision_rate = crash_counter / num_episodes
    # mean_return = total_ep_return / num_episodes

    # print(f"Mean Episode Length: {mean_ep_length}")
    # #print(f"Mean Speed: {mean_speed}")
    # print(f"Collision Rate: {collision_rate}")
    # print(f"Mean Return: {mean_return}")
    env.close()

    csv_id = 'test'
    metricObj.printRecap(variant['eval_output_dir'], csv_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='dm_env')
    parser.add_argument('--dataset', type=str, default=r'notebooks/train_eval/50000_dataset.npy')  # path to dataset
    parser.add_argument('--model', type=str, default=r'.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--eval_output_dir', type=str, default=r'.')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--K', type=int, default=35)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--video_folder', type=str, default=r'./videos/')

    args = parser.parse_args()
    evaluate_model(variant=vars(args))
