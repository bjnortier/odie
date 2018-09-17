# -*- coding: utf-8 -*-

import os, sys
from gym import Wrapper
from gym.wrappers import Monitor
from gym.wrappers.monitoring import video_recorder
import requests
from dotenv import load_dotenv, find_dotenv
from git import Repo
import numpy as np
from baselines import logger
from termcolor import colored

load_dotenv(find_dotenv('.minotaur'))
jwt = os.getenv("MINOTAUR_JWT")
assert jwt
minotaur_url = 'https://minotaur-1.herokuapp.com'

current_minotaur_experiment_id = None

def create_minotaur_experiment(env_id):
    repo = Repo(os.getcwd())
    is_dirty = repo.is_dirty()
    is_master = repo.refs.master.commit.hexsha == repo.head.reference.commit.hexsha
    is_master_synced = repo.refs.master.commit.hexsha == repo.remotes.origin.refs.master.commit.hexsha
    headers = {'Authorization': 'Bearer {0}'.format(jwt)}
    experiment_spec = {
        'name': env_id,
        'meta': {},
        'repo': list(repo.remotes.origin.urls)[0],
        'commit': repo.head.reference.commit.hexsha,
        'isClean': not is_dirty,
        'isMaster': is_master,
        'isMasterSynced': is_master_synced
    }
    response = requests.post('{}/api/experiments'.format(minotaur_url), headers=headers, json=experiment_spec)
    print('created experiment: "{0}"'.format(response.json()['name']))
    global current_minotaur_experiment_id
    current_minotaur_experiment_id = response.json()['id']

baselines_logger_logkv = logger.logkv
baselines_logger_dumpkvs = logger.dumpkvs

log_values = {}

def minotaur_logkv(key, value):
    global log_values
    if hasattr(value, 'dtype'):
        log_values[key] = value.item()
    else:
        log_values[key] = value
    baselines_logger_logkv(key, value)

def minotaur_dumpkvs():
    global log_values
    global current_minotaur_experiment_id
    response = requests.post(
        '{}/api/data/{}'.format(minotaur_url, current_minotaur_experiment_id),
        headers={'Authorization': 'Bearer {0}'.format(jwt)},
        json=log_values)
    if response.status_code == 202:
        print(colored('✓', 'green'), 'scalars')
    else:
        print(colored('✗', 'red'), response.status_code)
    log_values = {}
    baselines_logger_dumpkvs()

logger.logkv = minotaur_logkv
logger.dumpkvs = minotaur_dumpkvs

def video_schedule(episode_id):
    if (episode_id < 256):
        # Powers of 2
        return ((episode_id & (episode_id - 1)) == 0)
    else:
        # Multiples of 256
        return (episode_id & 0xff == 0)

class MinotaurMonitor(Wrapper):
    def __init__(self, env_id, env):
        super().__init__(env)
        global current_minotaur_experiment_id
        self.experiment_id = current_minotaur_experiment_id
        self.env_id = env_id
        self.episode_id = 1
        self.video_callable = video_schedule
        # self.episode_rewards = []
        self.headers = {'Authorization': 'Bearer {0}'.format(jwt)}
        self.create_video_recorder()

    def reset(self, **kwargs):
        try:
            self.video_recorder.close()
            filename = '{}.mp4'.format(self.base_path)
            if self.video_recorder.enabled and os.path.isfile(filename):
                response = requests.post(
                   '{}/api/data/{}'.format(minotaur_url, self.experiment_id),
                   headers=self.headers,
                   data={'episode': self.episode_id},
                   files={'animation': open(filename, 'rb')})
                if response.status_code == 202:
                    print(colored('✓', 'green'), 'video')
                else:
                    print(colored('✗', 'red'), response.status_code)
        except:
            print(colored('[ERR]', 'red'), sys.exc_info())
        self.episode_id += 1
        self.create_video_recorder()
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if not done:
            self.video_recorder.capture_frame()
        return observation, reward, done, info

    def create_video_recorder(self):
        self.base_path = os.path.join('/tmp', '{}.video{}'.format(self.env_id, self.episode_id))
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=self.base_path,
            metadata={'episode_id': self.episode_id},
            enabled=self.video_callable(self.episode_id)
        )

    def close(self):
        self.video_recorder.close()

    def video_enabled(self):
        return self.video_callable(self.episode_id)
