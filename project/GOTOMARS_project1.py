#!/usr/bin/env python2
from __future__ import print_function
from copy import deepcopy
import sys
import os
import rospkg
import rospy

######################################## PLEASE CHANGE TEAM NAME ########################################
TEAM_NAME = "GOTOMARS"
######################################## PLEASE CHANGE TEAM NAME ########################################
project_path = rospkg.RosPack().get_path("sim2real")
yaml_file = project_path + "/config/eval1.yaml"
PATH = project_path + "/scripts"
sys.path.append(PATH)

from sim2real.msg import Result, Query

import gym
import env
import numpy as np
import math
import yaml
import time


def dist(waypoint, pos):
    return math.sqrt((waypoint[0] - pos.x) ** 2 + (waypoint[1] - pos.y) ** 2)

class PurePursuit:
    def __init__(self, args):
        rospy.init_node(TEAM_NAME + "_project1", anonymous=True, disable_signals=True)

        # env reset with world file
        self.env = gym.make('RCCar-v0')
        self.env.seed(1)
        self.env.unwrapped
        self.env.load(args['world_name'])
        self.track_list = self.env.track_list
        self.time_limit = 100.0

        ######################################## YOU CAN ONLY CHANGE THIS PART ########################################
        # TO DO
        """ 
        Setting hyperparameter
        Recommend tuning PID coefficient P->D->I order.
        Also, Recommend set Ki extremely low.
        """
        self.lookahead = 10
        self.prv_err = None
        self.cur_err = None
        self.sum_err = 0.0
        self.dt = 0.1
        ##### implement for best ride #####
##<<<<<<< HEAD
        self.min_vel = 1.5
        self.max_vel = 3.0
        self.Kp = 3.0
        self.Ki = 0.001
        self.Kd = 1.1
##=======
      #  self.min_vel =8.0
     #   self.max_vel = 12.5
      #  self.Kp = 2.0
      #  self.Ki = 0.00008
      #  self.Kd = 1.0
##>>>>>>> 616b615889f49ca0936409c52cb508bf2d18d3b4
        ###################################
        ### For project 2 ###
        self.expert_data = {"state":[], "action":[], "reward":[]}
        #####################
        ######################################## YOU CAN ONLY CHANGE THIS PART ########################################

        self.query_sub = rospy.Subscriber("/query", Query, self.callback_query)
        self.rt_pub = rospy.Publisher("/result", Result, queue_size = 1)

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
            self.env.reset(name = data.world)
            self.waypoints = self.env.waypoints_list[self.env.track_id]
            self.N = len(self.waypoints)
            self.prv_err = None
            self.cur_err = None
            self.sum_err = 0.0

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
                

                cur_pos = self.env.get_pose()
                
                ######################################## YOU CAN ONLY CHANGE THIS PART ########################################
                # TO DO
                """ 
                1) Find nearest waypoint from a race-car
                2) Calculate error between the race-car heading and the direction vector between the lookahead waypoint and the race-car
                3) Determine input steering of the race-car using PID controller
                4) Calculate input velocity of the race-car appropriately in terms of input steering
                """

                # 1) nearest waypoint
                for i in range(self.N):
                    if i == 0 :
                        min_dist = dist(np.array(self.waypoints[0]), cur_pos.position)
                        nearest_index = 0
                    wp_rc_dist = dist(np.array(self.waypoints[i]), cur_pos.position)
                    if wp_rc_dist < min_dist:
                        min_dist = wp_rc_dist
                        nearest_index = i
                # lookahead waypoint
                lookahead_waypoint = self.waypoints[(nearest_index + self.lookahead)%len(self.waypoints)]

                if self.cur_err == None:
                    race_car_dir = np.array([0.0, 1.0])
                else:
                    race_car_dir = np.array([cur_pos.position.x, cur_pos.position.y]) - np.array([prv_pos.position.x, prv_pos.position.y])
                waypoint_dir = np.array(lookahead_waypoint) - np.array([cur_pos.position.x, cur_pos.position.y])
                
                # 2) error calculation
                self.prv_err = self.cur_err if self.cur_err != None else 0.0
                sign = -1 if np.cross(waypoint_dir, race_car_dir)<0 else 1
                pw_product = waypoint_dir[0]*race_car_dir[0] + waypoint_dir[1]*race_car_dir[1]
                self.cur_err = sign * np.arccos(pw_product/(np.linalg.norm(waypoint_dir)*np.linalg.norm(race_car_dir)))
                self.sum_err += self.cur_err

                # 3) 4) action change
		#### PID #################################
                input_steering = self.Kp*self.cur_err + self.Ki*self.dt*self.sum_err + self.Kd*(self.cur_err-self.prv_err)/self.dt
		#####################################
                if input_steering > 1.5: 
			input_steering = 1.5
                elif input_steering < -1.5: 
			input_steering = -1.5
                input_vel = self.min_vel + (self.max_vel - self.min_vel)*np.exp(-abs(input_steering))
                a = [input_steering, input_vel]
                prv_pos = cur_pos

                ### For project 2 ###
      		s,r,done,logs=self.env.step(a)
		print(s.size)
		print(r.size)
		self.expert_data["state"].append(s)
		self.expert_data["action"].append(a)
		self.expert_data["reward"].append(r)
                #####################

                ######################################## YOU CAN ONLY CHANGE THIS PART ########################################


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
                    if logs['info'] == 1:
                        rt.fail_type = "Collision"
                    if logs['info'] == 2:
                        rt.fail_type = "Exceed Time Limit"
                    self.rt_pub.publish(rt)

                    ### For project 2 ###
                    # flattening from dict to array
                    element = []
		    print(self.expert_data["state"][0])
		    print(len(self.expert_data["state"]))
                    for i in range(len(self.expert_data["state"])):
                        temp = []
                        for j in range(len(self.expert_data["state"][i])):
                            temp.append(self.expert_data["state"][i][j])
                        temp.append(self.expert_data["action"][i][0])   # steering
                        temp.append(self.expert_data["action"][i][1])   # velocity
                        temp.append(self.expert_data["reward"][i])
                        element.append(temp)
                    e = np.array(element)
                    # save data
                  


		    np.savetxt(project_path+'/project/IS_GOTOMARS/project/'+'map2_5', e)
		    #np.savetxt(project_path+'/project/IS_GOTOMARS/project/'+'expert_data_'+rt.world+'_'+str(self.min_vel)+'&'+str(self.max_vel)+'.txt', e)
                    print("saved")
                    #####################

                    break
        except:
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
    PurePursuit(args)
    rospy.spin()

