#!/usr/bin/env python2
from __future__ import print_function


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
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler    # normalization
from joblib import dump, load


project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval1.yaml"

TEAM_NAME = "GOTOMARS"
# Code by Sungyoung Lee
team_path = project_path + "/project/IS_" + TEAM_NAME

class GaussianProcess:
    def __init__(self, args):
        rospy.init_node('gaussian_process_' + TEAM_NAME, anonymous=True, disable_signals=True)
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        self.track_list = self.env.track_list
        
        self.time_limit = 100.0
        
        """
        add your demonstration files with expert state-action pairs.
        you can collect expert demonstration using pure pursuit.
        you can define additional class member variables.
        """

        # 1.5 <= minVel <= maxVel <= 3.0
        self.maxAng = 1.5
        self.minVel = 1.5
        self.maxVel = 3.0
        
        self.demo_files = []

        self.obs_num = 1000

        self.demo_obs = []
        self.demo_act = []
        self.demo_rwd = []

        self.kernel = C(constant_value=1.0, constant_value_bounds=(1e-5, 10.0))* RBF(length_scale=0.5, length_scale_bounds=(1e-5, 10.0))

        
        self.gp = GaussianProcessRegressor(kernel = self.kernel, alpha = 0.2)
        self.gp_file_name = "demo"
        self.gp_file = project_path + "/" + self.gp_file_name + ".joblib"

        self.load()

        self.query_sub = rospy.Subscriber("/query", Query, self.callback_query)
        self.rt_pub = rospy.Publisher("/result", Result, queue_size = 1)

        print("completed initialization")


    def load(self):
        """
        1) load your expert demonstration
        2) normalization and gp fitting (recommand using scikit-learn package)
        3) if you already have a fitted model, load it instead of fitting the model again.
        Please implement loading pretrained model part for fast evaluation.
        """
        if False:   # TODO: if fitted model exist
            1
        else:
            # 1) load expert demonstration
            track1_1 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_1_0.5&5.0.txt')
            track1_2 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_1_1.0&4.5.txt')
            track1_3 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_1_1.2&4.0.txt')
            track1_4 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_1_1.5&3.5.txt')
            track1_5 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_1_0.0&5.5.txt')
            track2_1 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_2_0.5&5.0.txt')
            track2_2 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_2_1.0&4.5.txt')
            track2_3 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/expert_data_track_2_1.2&4.0.txt')
            track1_total = np.append(track1_1, track1_2, axis=0)
            track1_total = np.append(track1_total, track1_3, axis=0)
            track1_total = np.append(track1_total, track1_4, axis=0)
            track1_total = np.append(track1_total, track1_5, axis=0)                                   
            track2_total = np.append(np.append(track2_1, track2_2, axis=0), track2_3, axis=0)
            
            self.demo_files = shuffle(np.append(track1_total, track2_total, axis=0)) # shuffle state action pairs
            # self.demo_files = shuffle(track1_total)
            self.demo_obs = np.clip(self.demo_files[:,:-5], 0, 30)  # prevent inf value
            self.demo_obs_action = np.array(self.demo_files[:,-5:-3])
            self.demo_act = np.array(self.demo_files[:,-3:-1])
            self.demo_rwd = np.array(self.demo_files[:,-1])
            print(np.shape(self.demo_obs))
            print(np.shape(self.demo_obs_action))
            print(np.shape(self.demo_act))
            print(np.shape(self.demo_rwd))

            ### downsizeing obs : 1083 -> 28 ###
            downsize_obs = []
            for i in range(len(self.demo_obs)):
                raw_obs = []
                for j in range(51):
                    raw_obs.append(np.mean(self.demo_obs[i][20*j:20*j+81]))
                # raw_obs.extend([self.demo_obs_action[i][0], self.demo_obs[i][1]])
                downsize_obs.append(raw_obs)
            ###------------------------------###  
            
            # 2) normalize
            # obs
            self.normalized_obs = StandardScaler()
            self.normalized_act = StandardScaler()
            # self.normalized_obs.fit(downsize_obs)
            self.normalized_obs.fit(self.demo_obs)
            self.normalized_act.fit(self.demo_act)
            # s = self.normalized_obs.transform(downsize_obs)
            s = self.normalized_obs.transform(self.demo_obs)
            a = self.normalized_act.transform(self.demo_act)

            # 3) gp fit   
            print("d : ", np.shape(s))
            print("a : ", np.shape(a))
            self.gp.fit(s, a)
            print(self.gp.score(s, a))
            print("fitted")
            print("")



    def get_action(self, obs):
        """
        1) input observation is the raw data from environment.
        2) 0 to 1080 components (totally 1081) are from lidar.
           Also, 1081 component and 1082 component are scaled velocity and steering angle, respectively.
        3) To improve your algorithm, you must process the raw data so that the model fits well.
        4) Always keep in mind of normalization process during implementation.
        """

        obs_clip = np.clip(obs[:,:-2], 0, 30)
        ### downsizeing obs : 1083 -> 28 ###
        raw_obs = []
        for i in range(51):
            raw_obs.append(np.mean(obs_clip[0][20*i:20*i+81]))
        # raw_obs.extend([obs[0][-2], obs[0][-1]])
        ###------------------------------###

        # normalize
        # s = self.normalized_obs.transform([raw_obs])
        s = self.normalized_obs.transform(obs_clip)
        [[angle_act, linvel_act]] = self.gp.predict(s)
        i_steering, i_linear = self.normalized_act.inverse_transform([angle_act, linvel_act])
        print("lin_actvel : ", i_linear)
        print("ang_actvel : ", i_steering)
        print(obs_clip)
        print("")
        return [[i_steering, i_linear]]



    def callback_query(self, data):
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
            obs = np.reshape(obs, [1,-1])
            
            
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
                
                act = self.get_action(obs)
                input_steering = np.clip(act[0][0], -self.maxAng, self.maxAng)
                input_velocity = np.clip(act[0][1], self.minVel, self.maxVel)
                obs, _, done, logs = self.env.step([input_steering, input_velocity])
                obs = np.reshape(obs, [1,-1])
                
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
    GaussianProcess(args)
    rospy.spin()

