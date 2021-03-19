# rl-benchmarks

w.i.p. report https://hackmd.io/@92tLxRFMRF-iTbw8_IQU6Q/Sk6aw_wjP/edit

FREEE SPACE: /local00/student/buzatu 

# tutorial setting up baselines
https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/

### Needed
- Python==3.7
- Ubuntu==18
- 

### Steps to set up this repo
```angular2html
git clone --verbose --recursive https://github.com/ioneliabuzatu/rl-benchmarks

python3 -m venv benchmarksVenv
source benchmarksVenv/bin/activate
pip3 install -r requirements

pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

source /system/apps/biosoft/OpenMPI/bashrc
```

```angular2html 
conda env create -f conda_environment.yml
conda activate benchmarksVenv37

# install torch for cuda
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch


conda install -c hcc pybullet

```


```angular2html
cd baselines && pip install -e .
pip install pytest  
```


```angular2html


# set up mujoco
mkdir ~/.mujoco/
mv mujoco200_linux .mujoco/mujoco200
mv mjkey.txt .mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/system/user/buzatu/.mujoco/mujoco200/bin
pip install mujoco
```


### TODO models benchmark

1. - [X] DDPG - [paper](https://arxiv.org/abs/1509.02971), [code]()
2. - [X] TD3 - [paper](https://spinningup.openai.com/en/latest/algorithms/td3.html), [code]()
3. - [X] SAC - [paper](https://arxiv.org/pdf/1801.01290.pdf), [code]()
4. - [X] PPO2 - [paper]()[code]()
5. - [X] TRPO - []()[]()   


### Experiments #1 with  MLP policy

|        | **Pendulum-v0** | ReacherBulletEnv-v0 | Hopper-v2 | Humanoid-v2 | HalfCheetah-v2 | HumanoidStandup-v2  |
| ------ | :-------------: | :-----------------: | :-------: | :----------:| :------------: | :-----------------: | 
| DDPG   | X               | X                   | X         | X           | X              | X                   |
| PPO2   | X               | X                   | X         | X           | X              | X                   | 
| SAC    | X               | X                   | X         | X           | X              | X                   | 
| TD3    | X               | X                   | X         | X           | X              | X                   | 
| TRPO   | 



### TODO benchmark environments 

1. - [ ] [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0)
1. - [ ] [Humanoid-v2](https://gym.openai.com/envs/Humanoid-v2)
1. - [ ] [Swimmer-v2](https://gym.openai.com/envs/Swimmer-v2)
1. - [ ] [Walker2d-v2](https://gym.openai.com/envs/Walker2d-v2)
1. - [ ] [MountainCarContinuous-v0]()
1. - [ ] [Hopper-v2]() 
1. - [ ] [Humanoid-v2]()
1. - [ ] [Walker2DBulletEnv-v0]()
1. - [ ] [HumanoidStandup-v2]()
1. - [ ] [HalfCheetah-v2]()
1. - [ ] [Swimmer-v2]()

#### show results with [experiment_buddy](https://github.com/ministry-of-silly-code/experiment_buddy)


```
.
├── baselines --> some openai models implementations I use for benchmark (list them) 
├── benchmark.py
├── DDPG --> my ddpg implementation
├── Practical Work in AI report.md --> 
├── README.md
├── requirements.py
├── softlearning --> repo for SAC model I use
└── TD3 --> repo for TD3 I use

```
