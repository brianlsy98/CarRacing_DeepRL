#!/usr/bin/env python3
from __future__ import print_function

# Change python2 to python3 above if you want to use PyTorch!

##### add python path #####
import sys
import os
import rospkg
import rospy

PATH = rospkg.RosPack().get_path("sim2real") + "/scripts"
print(PATH)
sys.path.append(PATH)


import gym
import env
import numpy as np
from collections import deque
import random
import math
import yaml
import time


from ppo_torch import Agent
from utils import plot_learning_curve

import torch


project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval12.yaml"

######################################## PLEASE CHANGE TEAM NAME ########################################
TEAM_NAME = "GOTOMARS"
######################################## PLEASE CHANGE TEAM NAME ########################################
team_path = project_path + "/project/IS_" + TEAM_NAME

class RunProject_3:
    def __init__(self):
        
        self.time_limit = 150.0

        self.maxAng = 1.5
        self.minVel = 1.5
        self.maxVel = 3.0

        self.load()

        print("completed initialization")


    def load(self):
        # == PPO Model == #
        alpha = 0.0003
        n_epochs = 4
        batch_size = 500
        self.agent = Agent(n_actions=11, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=(28,), path=team_path)
        
        # ============================================================================ #
        # self.agent.actor.load_checkpoint(self.agent.model_path+"actor_torch_ppo_final_1")
        # self.agent.critic.load_checkpoint(self.agent.model_path+"critic_torch_ppo_final_1")
        # self.agent.actor.load_checkpoint(self.agent.model_path+"actor_torch_ppo_final_3")
        # self.agent.critic.load_checkpoint(self.agent.model_path+"critic_torch_ppo_final_3")
        self.agent.actor.load_checkpoint(self.agent.model_path+"actor_torch_ppo_final_fast")
        self.agent.critic.load_checkpoint(self.agent.model_path+"critic_torch_ppo_final_fast")
        # ============================================================================ #

        # self.agent.actor.load_checkpoint(self.agent.model_path+"actor_torch_ppo")
        # self.agent.critic.load_checkpoint(self.agent.model_path+"critic_torch_ppo")
        
        print("PPO model loaded")
        
        return


    def fine_control(self, fc):

        if fc == 0: ang = -1.5
        elif fc == 1: ang = -1.2
        elif fc == 2: ang = -0.9
        elif fc == 3: ang = -0.6
        elif fc == 4: ang = -0.3
        elif fc == 5: ang = 0.0
        elif fc == 6: ang = 0.3
        elif fc == 7: ang = 0.6
        elif fc == 8: ang = 0.9
        elif fc == 9: ang = 1.2
        elif fc == 10: ang = 1.5

        return ang


    def obs_normalize(self, obs_high):   # from 1083 -> 26 + 2
        obs_prv_action = []
        obs_lidar = []
        
        for i in range(26):
            obs_lidar.append(np.mean(obs_high[40*i:40*i+81]))
        obs_prv_action.extend([obs_high[1082], obs_high[1081]])
        
        # normalize -> element sum to 1
        obs_low_lidar = np.array(obs_lidar)/np.sum(obs_lidar)
        obs_low_prvact = np.array(obs_prv_action)

        return np.concatenate((obs_low_lidar, obs_low_prvact)), obs_low_lidar, obs_low_prvact


    def callback_query(self, data):

        rt = Result()
        START_TIME = rospy.get_time()
        is_exit = data.exit
        try:
            # if query is valid, start
            if data.name != TEAM_NAME:
                return
            
            if data.world not in self.track_list:
                END_TIME = rospy.get_time()
                rt.id = data.id
                rt.trial = data.trial
                rt.team = data.name
                rt.world = data.world
                rt.elapsed_time = END_TIME - START_TIME
                rt.waypoints = 0
                rt.n_waypoints = 20
                rt.success = False
                rt.fail_type = "Invalid Track"
                self.rt_pub.publish(rt)
                return
            
            print("[%s] START TO EVALUATE! MAP NAME: %s" %(data.name, data.world))
            
            
            # ====== Initialization  ====== #
            obs = self.env.reset(name = data.world)
            obs = np.reshape(obs, [1,-1])
            obs[obs == float('inf')] = 10
            # ** for PPO model ** #
            obs_norm, _, _ = self.obs_normalize(obs[0])
            # ******************* #
            # ============================= #
            rsum = 0
            while True:
                if rospy.get_time() - START_TIME > self.time_limit:
                    END_TIME = rospy.get_time()
                    rt.id = data.id
                    rt.trial = data.trial
                    rt.team = data.name
                    rt.world = data.world
                    rt.elapsed_time = END_TIME - START_TIME
                    rt.waypoints = self.env.next_checkpoint
                    rt.n_waypoints = 20
                    rt.success = False
                    rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)
                    print("EXCEED TIME LIMIT")
                    break
                
                action, _, _ = self.agent.choose_action(obs_norm)
                # == Controls from PPO == #
                input_steering = self.fine_control(action)
                input_velocity = (3.0-abs(input_steering))
                # ======================= #

                obs_, reward, done, logs = self.env.step([input_steering, input_velocity])
                obs_ = np.reshape(obs_, [1,-1])
                obs_[obs_ == float('inf')] = 10
                # ** for PPO model ** #
                obs_norm_, _, _ = self.obs_normalize(obs_[0])
                # ******************* #
                obs_norm = obs_norm_

                rsum += reward

                if done:
                    END_TIME = rospy.get_time()
                    rt.id = data.id
                    rt.trial = data.trial
                    rt.team = data.name
                    rt.world = data.world
                    rt.elapsed_time = END_TIME - START_TIME
                    rt.waypoints = logs['checkpoints']
                    rt.n_waypoints = 20
                    rt.success = True if logs['info'] == 3 else False
                    rt.fail_type = ""
                    print("Time : "+str(rt.elapsed_time))
                    print("total reward : "+str(rsum))
                    print(logs)
                    if logs['info'] == 1:
                        rt.fail_type = "Collision"
                    if logs['info'] == 2:
                        rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)
                    print("publish result")
                    break
        
        except Exception as e:
            print(e)
            END_TIME = rospy.get_time()
            rt.id = data.id
            rt.trial = data.trial
            rt.team = data.name
            rt.world = data.world
            rt.elapsed_time = END_TIME - START_TIME
            rt.waypoints = 0
            rt.n_waypoints = 20
            rt.success = False
            rt.fail_type = "Script Error"
            self.rt_pub.publish(rt)

        if is_exit:
            rospy.signal_shutdown("End query")
        
        return

if __name__ == '__main__':
    env = gym.make('RCCar-v0')
    env.seed(1)
    env = env.unwrapped
    agent = RunProject_3()

    while(1):
        print("There's only final_1, final_3, final_fast, and just torch_ppo files in race car (total 4*2 = 8)")
        print("1: racing")
        print("2: break")
        opt = input("select option: ")
        opt = int(opt)
        if opt != 1:
            break
        # ====== Initialization  ====== #
        obs = env.reset()
        obs = np.reshape(obs, [1,-1])
        obs[obs == float('inf')] = 10
        # ** for PPO model ** #
        obs_norm, _, _ = agent.obs_normalize(obs[0])
        # ******************* #
        # ============================= #
        while(1):
            # == Controls from PPO == #
            action, _, _ = agent.agent.choose_action(obs_norm)
            input_steering = agent.fine_control(action)
            input_velocity = (3.0-abs(input_steering))
            # ======================= #

            obs_, reward, done, logs = agent.env.step([input_steering, input_velocity])
            obs_ = np.reshape(obs_, [1,-1])
            obs_[obs_ == float('inf')] = 10
            # ** for PPO model ** #
            obs_norm_, _, _ = agent.obs_normalize(obs_[0])
            # ******************* #
            obs_norm = obs_norm_
            if done:
                break








    rospy.spin()

