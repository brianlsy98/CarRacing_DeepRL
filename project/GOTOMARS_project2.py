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
from sklearn.utils import shuffle, resample
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

        self.obs_num = 500

        self.demo_obs = []
        self.demo_act = []
        self.demo_rwd = []

        self.kernel = C(constant_value=1.0, constant_value_bounds=(1e-5, 10.0))* RBF(length_scale=1.0, length_scale_bounds=(1e-5, 10.0)) + C(constant_value = 0.5, constant_value_bounds = (1e-5, 1.0)) * RBF(length_scale = 1.0, length_scale_bounds = (1e-5, 10.0))
        self.gp = GaussianProcessRegressor(kernel = self.kernel, alpha = 0.2)
        self.gp_file_name = "demo"
        self.gp_file = team_path + "/" + self.gp_file_name + ".joblib"
        self.obs_file = team_path + "/obs_file.pkl"
        self.act_file = team_path + "/act_file.pkl"

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
        try:   # if fitted model exist
            self.gp = load(self.gp_file)
            self.normalized_obs = load(self.obs_file)
            self.normalized_act = load(self.act_file)
            print('pre-trained model exists')
        except:
            print('no pre-trained model exists, start loading data')
            # 1) load expert demonstration
            track1_1 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map1_1')
            track1_2 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map1_2')
            track1_3 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map1_3')
            track1_4 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map1_4')
            track1_5 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map1_5')
            track2_1 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map2_1')
            track2_2 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map2_2')
            track2_3 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map2_3')
            track2_4 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map2_4')
            track2_5 = np.loadtxt(project_path+'/project/IS_GOTOMARS/project/map2_5')
            
            track1_total = np.append(track1_1, track1_2, axis=0)
            track1_total = np.append(track1_total, track1_3, axis=0)
            track1_total = np.append(track1_total, track1_4, axis=0)
            track1_total = np.append(track1_total, track1_5, axis=0)                                   
            track2_total = np.append(np.append(track2_1, track2_2, axis=0), track2_3, axis=0)
            track2_total = np.append(track2_total,track2_4,axis=0)
            track2_total = np.append(track2_total,track2_5,axis=0)
            self.demo_files = np.append(track1_total, track2_total, axis=0)
            self.demo_obs = np.clip(self.demo_files[:, :-5], 0, 30)     # prevent inf value
            self.demo_obs_action = np.array(self.demo_files[:,-5:-3])
            self.demo_act = np.array(self.demo_files[:,-3:-1])
            self.demo_rwd = np.array(self.demo_files[:,-1])
            self.demo_obs = self.demo_obs.tolist()
            self.demo_act = self.demo_act.tolist()

            for i in range(len(self.demo_obs)):                         # append more datas for streering to the right side properly
                if self.demo_obs[i][0] > 0:
                    self.demo_obs.append(self.demo_obs[i])
                    self.demo_obs.append(self.demo_obs[i])
                    self.demo_obs.append(self.demo_obs[i])
                    self.demo_act.append(self.demo_act[i])
                    self.demo_act.append(self.demo_act[i])
                    self.demo_act.append(self.demo_act[i])
            self.demo_act = np.array(self.demo_act)
            self.demo_obs = np.array(self.demo_obs)

            print(np.shape(self.demo_obs))
            print(np.shape(self.demo_obs_action))
            print(np.shape(self.demo_act))
            print(np.shape(self.demo_rwd))
            

            # 2) normalize
            self.demo_obs, self.demo_act = resample(self.demo_obs, self.demo_act, n_samples = self.obs_num, replace = False)
            self.normalized_obs = StandardScaler()
            self.normalized_act = StandardScaler()
            self.normalized_obs.fit(self.demo_obs)
            self.normalized_act.fit(self.demo_act)
            transformed_obs = self.normalized_obs.transform(self.demo_obs)
            transformed_act = self.normalized_act.transform(self.demo_act)
            dump(self.normalized_obs, self.obs_file)
            dump(self.normalized_act, self.act_file)
            

            # 3) get gp fitted with normalized, shuffled datas
            s, a = shuffle(transformed_obs, transformed_act)
            print("d : ", np.shape(s))
            print("a : ", np.shape(a))
            self.gp.fit(s, a)
            print(self.gp.score(s, a))
            print("fitted")
            dump(self.gp, self.gp_file)

        return

        



    def get_action(self, obs):
        """
        1) input observation is the raw data from environment.
        2) 0 to 1080 components (totally 1081) are from lidar.
           Also, 1081 component and 1082 component are scaled velocity and steering angle, respectively.
        3) To improve your algorithm, you must process the raw data so that the model fits well.
        4) Always keep in mind of normalization process during implementation.
        """

        obs_clip = np.clip(obs[:,:-2], 0, 30)
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