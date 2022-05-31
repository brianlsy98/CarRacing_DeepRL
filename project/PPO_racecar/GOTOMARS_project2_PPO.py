#!/usr/bin/env python2

from __future__ import print_function
from imp import new_module


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
import json
import random
import math
import yaml
import time
from sim2real.msg import Result, Query


import tensorflow as tf
from tensorflow import keras
from PPO_keras import *
import matplotlib.pyplot as plt


project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval1.yaml"

TEAM_NAME = "GOTOMARS"
# Code by Sungyoung Lee
team_path = project_path + "/project/IS_" + TEAM_NAME

class ProximalPolicyOptimization:
    def __init__(self, args):
        rospy.init_node('gaussian_process_' + TEAM_NAME, anonymous=True, disable_signals=True)
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        print(args['world_name'])
        self.track_list = self.env.track_list
        
        ###### Select track ######
        self.track_name = 'track_1'
        ##########################

        self.time_limit = 100.0
        

        # 1.5 <= minVel <= maxVel <= 3.0
        self.maxAng = 1.5
        self.minVel = 1.5
        self.maxVel = 3.0

        self.hidden_sizes = (64, 64, 32)
        self.load()

        if self.train == 1:
            self.train_car()
        elif self.eval == 1:
            self.query_sub = rospy.Subscriber("/query", Query, self.callback_query_eval)

        self.rt_pub = rospy.Publisher("/result", Result, queue_size = 1)

        print("completed initialization")


    def load(self):
        """
        1) load your expert demonstration
        2) normalization and gp fitting (recommand using scikit-learn package)
        3) if you already have a fitted model, load it instead of fitting the model again.
        Please implement loading pretrained model part for fast evaluation.
        """
        self.train = 0; self.eval = 0; self.load_and_train = 0
        print("")
        if os.path.exists(project_path+'/project/IS_GOTOMARS/project/saved_models/PPO/'+str(self.track_name)+'/PPO_actor_weights.index'):
            ########## Loading Actor & Critic Keras Model (weights) ##########
            observation_input = keras.Input(shape=(28,), dtype=tf.float32)
            logits = mlp(observation_input, list(self.hidden_sizes) + [7], tf.tanh, None)
            self.my_actor = keras.Model(inputs=observation_input, outputs=logits)
            self.my_actor.load_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/PPO/'+str(self.track_name)+'/PPO_actor_weights')
            value = tf.squeeze(
                mlp(observation_input, list(self.hidden_sizes) + [1], tf.tanh, None), axis=1
            )
            self.my_critic = keras.Model(inputs=observation_input, outputs=value)
            self.my_critic.load_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/PPO/'+str(self.track_name)+'/PPO_critic_weights')
                
            print("")
            INpUT = input("load and train(1) or evaluate(2)? ")
            print("")

            if INpUT == 1:
                print("load and train mode")
                self.load_and_train = 1
                self.train = 1

            if INpUT == 2:
                print("evaluation mode")
                # evaluation mode
                self.eval = 1
                
        else:
            print("")
            print("training mode")
            # training mode
            self.train = 1

    def obs_dim_change(self, obs_high):   # from 1083 -> 28
        obs_prv_action = []
        obs_lidar = []
        obs_low = []
        for i in range(len(obs_high[0])-2):
            obs_lidar.append(np.clip(obs_high[0][i], 0, 10))
        obs_prv_action.extend([obs_high[0][-2], obs_high[0][-1]])
        for i in range(26):
             obs_low.append(np.mean(obs_lidar[4*i:4*i+80]))
        obs_low.extend(obs_prv_action)
        return obs_low



    def train_car(self):
        print("[%s] START TO TRAIN! MAP NAME: %s" %("GOTOMARS", "track 1 & 2"))
            
        ##### PPO object #####
        PPO = PPO_keras()
        PPO.set_layers(self.hidden_sizes)
        if self.load_and_train == 1:
            PPO.actor = self.my_actor
            PPO.critic = self.my_critic
        ######################
        # main iteration
        track_name = self.track_name
        PPO.pretrain_withBC(track_name)
        PPO.train(track_name)

        Actor_for_save = PPO.actor
        Critic_for_save = PPO.critic
        Actor_for_save.save_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/PPO/'+str(track_name)+'/PPO_actor_weights')
        Critic_for_save.save_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/PPO/'+str(track_name)+'/PPO_critic_weights')
        
        return





    def callback_query_eval(self, data):
        rt = Result()
        START_TIME = time.time()
        is_exit = data.exit
        try:
            # if query is valid, start
            if data.name != TEAM_NAME:
                return
            
            if data.world not in self.track_list:
                END_TIME = time.time()
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
            obs = self.env.reset(name = data.world)
            

            ####### Define PPO Model #######
            model = PPO_keras()
            model.load_actor(self.my_actor)
            model.load_critic(self.my_critic)
            ################################

            rsum = 0

            while True:
                if time.time() - START_TIME > self.time_limit:
                    END_TIME = time.time()
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
                
                obs = obs.reshape(1, -1)
                obs_dim217 = self.obs_dim_change(obs)
                _, action = model.sample_action(np.array(obs_dim217).reshape(1, -1))
                input_steering, input_velocity = model.action_to_angle_and_vel(action)
                obs_new, reward, done, logs = self.env.step([input_steering, input_velocity])
                rsum += reward
                print("")
                print("lidar max : "+str(max(obs_dim217[:-2])))
                print("lidar min : "+str(min(obs_dim217[:-2])))
                print("input steer : "+str(input_steering))
                print("input vel : "+str(input_velocity))
                obs = obs_new
                
                if done:
                    END_TIME = time.time()
                    rt.id = data.id
                    rt.trial = data.trial
                    rt.team = data.name
                    rt.world = data.world
                    rt.elapsed_time = END_TIME - START_TIME
                    rt.waypoints = logs['checkpoints']
                    rt.n_waypoints = 20
                    rt.success = True if logs['info'] == 3 else False
                    rt.fail_type = ""
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
            END_TIME = time.time()
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
    ProximalPolicyOptimization(args)
    rospy.spin()

