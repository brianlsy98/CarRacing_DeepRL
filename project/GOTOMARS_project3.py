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
from sim2real.msg import Result, Query


from ppo_torch import Agent
from utils import plot_learning_curve

import torch


project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval3.yaml"

######################################## PLEASE CHANGE TEAM NAME ########################################
TEAM_NAME = "GOTOMARS"
######################################## PLEASE CHANGE TEAM NAME ########################################
team_path = project_path + "/project/IS_" + TEAM_NAME

class RunProject_3:
    def __init__(self, args):
        rospy.init_node('gaussian_process_' + TEAM_NAME, anonymous=True, disable_signals=True)
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        self.track_list = self.env.track_list
        
        self.time_limit = 150.0

        #############################
        #############################
        # ** load and train if 1 ** #
        self.load_and_train = 0
        # ************************* #
        #############################
        #############################


        """
        add your demonstration files with expert state-action pairs.
        you can collect expert demonstration using pure pursuit.
        you can define additional class member variables.
        """
        #DON'T CHANGE THIS PART!
        # 1.5 <= minVel <= maxVel <= 3.0
        self.maxAng = 1.5
        self.minVel = 1.5
        self.maxVel = 3.0
        ########################
        
        self.demo_files = []


        print("")
        self.load()

        self.query_sub = rospy.Subscriber("/query", Query, self.callback_query)
        self.rt_pub = rospy.Publisher("/result", Result, queue_size = 1)

        print("completed initialization")


    def load(self):
        # == PPO Model == #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alpha = 0.0003
        n_epochs = 4
        batch_size = 500
        self.agent = Agent(n_actions=11, batch_size=batch_size, 
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=(28,), path=team_path)
        
        try:
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
            if self.load_and_train == 1: self.agent.train(tracklist = ["track_10", "track_11", "track_13"], e = self.env)
        except:
            print("PPO model unloaded")
            self.agent.train(tracklist = self.track_list, e = self.env)
        # =============== #
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
    with open(yaml_file) as file:
        args = yaml.load(file)
    RunProject_3(args)
    rospy.spin()

