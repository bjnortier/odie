import os, sys
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
minotaur_url = 'https://minotaur-1.herokuapp.com'

def create_minotaur_experiment():
    repo = Repo(os.path.dirname(__file__))
    is_dirty = repo.is_dirty()
    is_master = repo.refs.master.commit.hexsha == repo.head.reference.commit.hexsha
    is_master_synced = repo.refs.master.commit.hexsha == repo.remotes.origin.refs.master.commit.hexsha
    headers = {'Authorization': 'Bearer {0}'.format(jwt)}
    experiment_spec = {
        'name': 'Odie-v2',
        'meta': {},
        'repo': list(repo.remotes.origin.urls)[0],
        'commit': repo.head.reference.commit.hexsha,
        'isClean': not is_dirty,
        'isMaster': is_master,
        'isMasterSynced': is_master_synced
    }
    response = requests.post('{}/api/experiments'.format(minotaur_url), headers=headers, json=experiment_spec)
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
        self.headers = {'Authorization': 'Bearer {0}'.format(jwt)}
        self.create_video_recorder()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.episode_rewards.append(reward)
        if done:
            try:
                # Upload the episode reward mean
                episode_total_reward = np.sum(np.array(self.episode_rewards))
                response = requests.post(
                    '{}/api/data/{}'.format(minotaur_url, self.experiment_id),
                    headers=self.headers,
                    json={
                        'episode': self.episode_id,
                        'episode_total_reward': episode_total_reward
                    })
                print('posting scalars: {}:{}'.format(response.status_code, response.content))
                # Upload the video recording
                self.video_recorder.close()
                if (self.video_callable(self.episode_id)):
                    filename = '{}.mp4'.format(self.base_path)
                    response = requests.post(
                       '{}/api/data/{}'.format(minotaur_url, self.experiment_id),
                       headers=self.headers,
                       data={'episode': self.episode_id},
                       files={'animation': open(filename, 'rb')})
                    print('posting video: {}:{}'.format(response.status_code, response.content))
            except:
                print('[ERR]', sys.exc_info())
            # Prepare for next episode
            self.episode_rewards = []
            self.episode_id += 1
            self.create_video_recorder()
        else:
            self.video_recorder.capture_frame()
        return observation, reward, done, info

    def post_data(self, data):
        try:
            data['episode'] = self.episode_id
            response = requests.post(
                '{}/api/data/{}'.format(minotaur_url, self.experiment_id),
                headers=self.headers,
                json=data)
            print('posting scalars: {}:{}'.format(response.status_code, response.content))
        except:
            print('[ERR]', sys.exc_info())

    def create_video_recorder(self):
        self.base_path = os.path.join('/tmp', '{}.video{}'.format(self.env_id, self.episode_id))
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.env,
            base_path=self.base_path,
            metadata={'episode_id': self.episode_id},
            enabled=self.video_callable(self.episode_id)
        )
        self.video_recorder.capture_frame()

    def close(self):
        self.video_recorder.close()

    def video_enabled(self):
        return self.video_callable(self.episode_id)
