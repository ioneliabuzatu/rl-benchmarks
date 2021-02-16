# rl-benchmarks

w.i.p. report https://hackmd.io/@92tLxRFMRF-iTbw8_IQU6Q/Sk6aw_wjP/edit

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

```

```angular2html 
conda env create -f conda_environment.yml
conda activate benchmarksVenv37

# install torch for cuda
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch
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

3. - [ ] DQN - [paper](https://arxiv.org/abs/1312.5602), [code]()
2. - [ ] DPG - [paper](https://deepmind.com/research/publications/deterministic-policy-gradient-algorithms), [code]()
1. - [X] DDPG - [paper](https://arxiv.org/abs/1509.02971), [code]()
4. - [X] TD3 - [paper](https://spinningup.openai.com/en/latest/algorithms/td3.html), [code]()
5. - [X] SAC - [paper](https://arxiv.org/pdf/1801.01290.pdf), [code]()

#  DDPG, GAIL, PPO1 and TRPO have issues in stable_baselines

| **Name**            | source code | **Recurrent**      | ```Box```          | ```Discrete```     | ```MultiDiscrete``` | ```MultiBinary```  | **Multi Processing**              |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| ACER                | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(5)</sup> | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| ACKTR               | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| DDPG                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: <sup>(4)</sup>|
| DQN                 | :heavy_check_mark:           | :x:                | :x:                | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| GAIL <sup>(2)</sup> | :heavy_check_mark:           | :x:                | :heavy_check_mark: |:heavy_check_mark:| :x:                 | :x:                | :heavy_check_mark: <sup>(4)</sup> |
| HER <sup>(3)</sup>  | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :heavy_check_mark:| :x:                               |
| PPO1                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |
| PPO2                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| SAC                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TD3                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TRPO                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |



### TODO benchmark environments 

1. - [ ] [CartPole-v1](https://gym.openai.com/envs/CartPole-v1)
2. - [ ] [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0)
1. - [ ] [Ant-v2](https://gym.openai.com/envs/Ant-v2)
1. - [ ] [Humanoid-v2](https://gym.openai.com/envs/Humanoid-v2)
1. - [ ] [Swimmer-v2](https://gym.openai.com/envs/Swimmer-v2)
1. - [ ] [Walker2d-v2](https://gym.openai.com/envs/Walker2d-v2)

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
