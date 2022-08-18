
when pushing to git :
    - check : self.load_and_train = 0
    - check : loading file name (in method load -> try:)

v1 -> v2 -> v3

stable models (expected)
: save when laptime is min
_torch_ppo_v0 : track 1, 2, 3 randomly choose and reinforce
_torch_ppo_v1 : track 1 reinforce   
_torch_ppo_v2 : track 2 reinforce   
_torch_ppo_v3 : track 3 reinforce
_torch_ppo_v6 : track 6 reinforce
_torch_ppo_v7 : track 7 reinforce        

_torch_ppo_slow : track 1 ~ 8 reinforce xxxxx not use
_torch_ppo_safemodel : all track completed once

###################################
_torch_ppo_best_v1 : safe model
_torch_ppo_fasttry : fast model     input vel only 1.5 when steering 1.5 else 3.0
###################################

_torch_ppo_v123-2.0 : track 1, 2, 3 with vel 2.0

_torch_ppo_vmin : track 6 reinforce

    -> 41초대는 내가볼 때 maxspeed로 모든 구간 움직임   (우리 최대기록은 44초)
_torch_ppo_v1_maxvel : track 1 reinforce with vel = maxvel (28초)
_torch_ppo_v3_maxvel : track 3 reinforce with vel = maxvel

######################
# 일어나서 쿼리 제출할 것들 :
 1) ppo_v1,v3_maxvel : GOTOMARS_project3에 inputvel maxVel로 바꾸고 load_and_train 0으로 push
 2) ppo_v123_2.0 : inputvel minVel

현재 깃에 올라간 버전이 v1_maxvel이고
로컬은 v123_2.0
######################

relatively unstable models (expected)
_torch_ppo_vx : reward -= 500 delete


map estimation:
- torch_ppo_v2 -> map 6 완주 : 2, 6 비슷한듯?
- 아직 map 10 남음...
- 골고루 한 후 track2 매우 조금 했을 때 6 10 빼고 다 완주
- torch_ppo_v3 -> map 3, 7, 9 개선 : 3, 7, 9 비슷한듯?  + 1도..