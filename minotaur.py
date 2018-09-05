import os
from gym import Wrapper
from gym.wrappers import Monitor
from gym.wrappers.monitoring import video_recorder
import requests
from dotenv import load_dotenv, find_dotenv
from git import Repo
import numpy as np

load_dotenv(find_dotenv('.minotaur'))
jwt = os.getenv("MINOTAUR_JWT")
assert jwt

def create_minotaur_experiment():
    repo = Repo(os.path.dirname(__file__))
    headers = {'Authorization': 'Bearer {0}'.format(jwt)}
    experiment_spec = {
        'name': 'Odie-v2',
        'meta': {},
        'repo': list(repo.remotes.origin.urls)[0],
        'commit': repo.head.reference.commit.hexsha
    }
    response = requests.post('http://localhost:8100/api/experiments', headers=headers, json=experiment_spec)
    print('created experiment: "{0}"'.format(response.json()['name']))
    return response.json()['id']

def power_of_two_video_schedule(episode_id):
    return episode_id != 0 and ((episode_id & (episode_id - 1)) == 0)

class MinotaurMonitor(Wrapper):
    def __init__(self, env_id, env):
        super().__init__(env)
        self.experiment_id = create_minotaur_experiment()
        self.env_id = env_id
        self.episode_id = 1
        self.video_callable = power_of_two_video_schedule
        self.episode_rewards = []
        self.create_video_recorder()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_rewards.append(reward)
        if done:
            headers = {'Authorization': 'Bearer {0}'.format(jwt)}
            # Upload the episode reward mean
            epsode_total_reward = np.sum(np.array(self.episode_rewards))
            response = requests.post(
                'http://localhost:8100/api/data/{0}'.format(self.experiment_id),
                headers=headers,
                json={'episode_total_reward': episode_total_reward})
            print('posting metrics: {}:{}'.format(response.status_code, response.content))
            # Upload the video recording
            self.video_recorder.close()
            filename = '{}.mp4'.format(self.base_path)
            response = requests.post(
               'http://localhost:8100/api/data/{0}'.format(self.experiment_id),
               headers=headers,
               files={'animation': open(filename, 'rb')})
            print('posting video: {}:{}'.format(response.status_code, response.content))
            # Prepare for next episode
            self.episode_rewards = []
            self.episode_id += 1
            self.create_video_recorder()
        else:
            self.video_recorder.capture_frame()
        return observation, reward, done, info

    def create_video_recorder(self):
        self.base_path = os.path.join('/tmp', '{}.video{}'.format(self.env_id, self.episode_id))
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=self.base_path,
            metadata={'episode_id': self.episode_id},
            enabled=True
        )
        self.video_recorder.capture_frame()

    def close(self):
        self.video_recorder.close()

    def video_enabled(self):
        return self.video_callable(self.episode_id)
