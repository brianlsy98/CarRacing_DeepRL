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
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler    # normalization
from joblib import dump, load

import tensorflow as tf
from tensorflow import keras
from myDQNclasses import Qfunction, DQN, ReplayBuffer
import matplotlib.pyplot as plt


project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval1.yaml"

TEAM_NAME = "GOTOMARS"
# Code by Sungyoung Lee
team_path = project_path + "/project/IS_" + TEAM_NAME

class DeepQNetwork:
    def __init__(self, args):
        rospy.init_node('gaussian_process_' + TEAM_NAME, anonymous=True, disable_signals=True)
        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        print(args['world_name'])
        self.track_list = self.env.track_list
        
        self.time_limit = 100.0
        
        """
        add your demonstration files with expert state-action pairs.
        you can collect expert demonstration using pure pursuit.
        you can define additional class member variables.
        """

        # 1.5 <= minVel <= maxVel <= 3.0
        self.maxAng = 1.5
        self.discrete_steering_angle_list = []
        for i in range(31): # input steering : 0, +-(0.1 ~ 1.5)
            self.discrete_steering_angle_list.append(-self.maxAng+i*self.maxAng/15)
        print(self.discrete_steering_angle_list)
        self.minVel = 1.5
        self.maxVel = 3.0
        
        self.obs_num = 1000

        self.load()

        if self.train == 1:
            # self.query_sub = rospy.Subscriber("/query", Query, self.callback_query_train)
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
        self.train = 0; self.eval = 0
        ############################################
        file_name = project_path + '/project/IS_GOTOMARS/project/saved_models/DQN/DQN_model_weights_5600'
        ############################################

        print("")
        if os.path.exists(file_name+'.index'):
            print("")
            print("evaluation mode")
            # evaluation mode
            self.eval = 1
            self.my_qfunction_model = Qfunction(28, 11, [512, 256, 128, 64])
            self.my_qfunction_model.load_weights(file_name)
        else:
            print("")
            print("training mode")
            # training mode
            self.train = 1
    



    def get_action(self, obs, eps, totalstep, initialize, Qprincipal):
        # epsilon greedy
        if np.random.rand() < eps or totalstep <= initialize:
            action = random.randint(0, len(self.discrete_steering_angle_list)-1)
            input_steering = self.discrete_steering_angle_list[action]
            # input_velocity = self.maxVel + abs(input_steering)*(self.minVel - self.maxVel)/self.maxAng
        else:
            Q = Qprincipal.compute_Qvalues(np.array(obs))
            action = np.argmax(Q)
            input_steering = self.discrete_steering_angle_list[action]
            # input_velocity = self.maxVel + abs(input_steering)*(self.minVel - self.maxVel)/self.maxAng
        return [input_steering, self.minVel], action
        


    def obs_dim_change(self, obs_high):   # from 1083 -> 28
        obs_prv_action = []
        obs_lidar = []
        obs_low = []
        for i in range(len(obs_high[0])-2):
            obs_lidar.append(np.clip(obs_high[0][i], 0, 10))
        obs_prv_action.extend([obs_high[0][-2], obs_high[0][-1]])
        for i in range(26):
            obs_low.append(np.mean(obs_lidar[40*i:40*i+80]))
        obs_low.extend(obs_prv_action)
        return obs_low



    def train_car(self):
        # rt = Result()
        # START_TIME = time.time()
        # is_exit = data.exit
        # try:
        #     # if query is valid, start
        #     if data.name != TEAM_NAME:
        #         return
            
        #     if data.world not in self.track_list:
        #         END_TIME = time.time()
        #         rt.id = data.id
        #         rt.trial = data.trial
        #         rt.team = data.name
        #         rt.world = data.world
        #         rt.elapsed_time = END_TIME - START_TIME
        #         rt.waypoints = 0
        #         rt.n_waypoints = 20
        #         rt.success = False
        #         rt.fail_type = "Invalid Track"
        #         self.rt_pub.publish(rt)
        #         return
            
            print("[%s] START TO TRAIN! MAP NAME: %s" %("GOTOMARS", "track_1"))
            ###########################################################################
            # HyperParameters
            lr = 5e-4  # learning rate for gradient update 
            batchsize = 128  # batchsize for buffer sampling
            maxlength = 10000  # max number of tuples held by buffer
            tau = 100  # time steps for target update
            episodes = 10000  # number of episodes to run
            initialize = 500  # initial time steps before start updating
            eps = 1
            eps_minus = .0001

            gamma = .99  # discount
            hidden_dims=[512, 256, 128, 64] # hidden dimensions
            ###########################################################################
            # setting before training
            obssize = 28    # 1081(lidar) + 2 -> 26 + 2
            actsize = 31    # input steering : 0, +-(0.1 ~ 1.5)
            optimizer = keras.optimizers.Adam(learning_rate=lr)

            Qprincipal = DQN(obssize, actsize, hidden_dims, optimizer)
            Qtarget = DQN(obssize, actsize, hidden_dims, optimizer)
            buffer = ReplayBuffer(maxlength)
            ###########################################################################
            # main iteration
            rrecord = []
            totalstep = 0

            for ite in range(episodes):
                # obs = self.env.reset(name = data.world)
                obs = self.env.reset(name = "track_1")
                obs = np.reshape(obs, [1,-1])
                obs_dim28 = self.obs_dim_change(obs)
                done = False
                rsum = 0
                env_reseted = 1
                while not done:
                    # counter <- counter +1
                    totalstep += 1

                    # epsilon greedy
                    if eps > 0.05 and totalstep > initialize: eps -= eps_minus
                    elif eps < 0.05 and totalstep > initialize: eps = 0.05
                    # action selection
                    action, action_index = self.get_action(obs_dim28, eps, totalstep, initialize, Qprincipal)                    
                    
                    if env_reseted == 1: action, action_index = [0, 0], 15
                    env_reseted = 0

                    prev_obs = obs_dim28
                    obs, reward, done, logs = self.env.step(action)
                    obs = np.reshape(obs, [1,-1])
                    obs_dim28 = self.obs_dim_change(obs)
                    
                    if np.min(obs_dim28[:-2]) < 0.6: # wall very close in front 
                        done = True

                    reward += action[1]

                    if done and logs['info']!=3 : reward -= 500

                    rsum += reward


                    # Save experience to Buffer
                    buffer.append((prev_obs, action_index, reward, obs_dim28, done))
                    
                    # Sample N tuples from replay buffer
                    if totalstep > initialize:
                        # [s, a, r, s', done] 64 pairs
                        SAMPLE = buffer.sample(batchsize)
                        # Compute target d_j for 1 <= j <= N
                        d = []
                        for j in range(len(SAMPLE)):
                            if not SAMPLE[j][4]:
                                # DQN
                                k = SAMPLE[j][2] + gamma*np.max(Qtarget.compute_Qvalues(np.array(SAMPLE[j][3])))
                            elif SAMPLE[j][4]:
                                k = SAMPLE[j][2]                            
                            d.append(k)
                    # Compute empirical loss & Update theta
                        array_s = np.array([SAMPLE[j][0] for j in range(len(SAMPLE))])
                        array_a = np.array([SAMPLE[j][1] for j in range(len(SAMPLE))])
                        l = Qprincipal.train(array_s, array_a, tf.convert_to_tensor(d, dtype=tf.float32))
                        
                    # Update target network theta_-
                    if totalstep % tau == 0:
                        print("")
                        print("epsilon : ", eps)
                        print("target updated, totalstep : ", totalstep)
                        Qtarget.update_weights(Qprincipal)
                    
                    pass

                
                # if done:
                #     END_TIME = time.time()
                #     rt.id = data.id
                #     rt.trial = data.trial
                #     rt.team = data.name
                #     rt.world = data.world
                #     rt.elapsed_time = END_TIME - START_TIME
                #     rt.waypoints = logs['checkpoints']
                #     rt.n_waypoints = 20
                #     rt.success = True if logs['info'] == 3 else False
                #     rt.fail_type = ""
                #     # print(logs)
                #     if logs['info'] == 1:
                #         rt.fail_type = "Collision"
                #     if logs['info'] == 2:
                #         rt.fail_type = "Exceed Time Limit"
                #     self.rt_pub.publish(rt)
                #     # print("publish result")
                #     START_TIME = time.time()
                

                rrecord.append(rsum)
                if ite % 10 == 0:
                    print('iteration {} ave reward {}'.format(ite, np.mean(rrecord[-10:])))
                    if ite % 200 == 0:
                        Qfunction_for_save = Qprincipal.qfunction
                        Qfunction_for_save.save_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/DQN/DQN_model_weights_'+str(ite))

            ###########################################################################
            # plot [episode, reward] history
            x = [i+1 for i in range(len(rrecord))]
            plt.plot(x, rrecord)
            plt.title('episode rewards')
            plt.xlabel('episodes')
            plt.ylabel('rewards')
            plt.show()

            Qfunction_for_save = Qprincipal.qfunction
            Qfunction_for_save.save_weights(project_path+'/project/IS_GOTOMARS/project/saved_models/DQN/DQN_model_weights')



        # except Exception as e:
        #     print(e)
        #     END_TIME = time.time()
        #     rt.id = data.id
        #     rt.trial = data.trial
        #     rt.team = data.name
        #     rt.world = data.world
        #     rt.elapsed_time = END_TIME - START_TIME
        #     rt.waypoints = 0
        #     rt.n_waypoints = 20
        #     rt.success = False
        #     rt.fail_type = "Script Error"
        #     self.rt_pub.publish(rt)

        # if is_exit:
        #     rospy.signal_shutdown("End query")
        
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
            obs = np.reshape(obs, [1,-1])
            obs_dim28 = self.obs_dim_change(obs)
            
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
                
                Q = self.my_qfunction_model(np.array([obs_dim28]))
                action = np.argmax(Q)
                input_steering = self.discrete_steering_angle_list[action]
                obs, _, done, logs = self.env.step([input_steering, self.maxVel])
                obs = np.reshape(obs, [1,-1])
                obs_dim28 = self.obs_dim_change(obs)
                
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
    DeepQNetwork(args)
    rospy.spin()

