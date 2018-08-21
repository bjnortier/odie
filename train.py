#!/usr/bin/env python3
# Adapted from baselines/ppo1/run_humanoid.py
import sys
sys.path.insert(0, "/Users/bjnortier/development/RL/gym")
sys.path.insert(0, "/Users/bjnortier/development/RL/baselines")
import os
import numpy as np
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U, zipsame
from baselines import logger
import gym
import gym.spaces
from gym.wrappers import Monitor
import requests
from dotenv import load_dotenv, find_dotenv

gym.envs.register(
    id='Odie-v2',
    entry_point='envs.odie_v2:OdieV2Env',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

load_dotenv(find_dotenv('.minotaur'))
jwt = os.getenv("MINOTAUR_JWT")
assert jwt
def create_minotaur_context():
    headers = {'Authorization': 'Bearer {0}'.format(jwt)}
    context_spec = {
        'name': 'Odie-v2',
        'x': {'key': 'i', 'label': 'Iterations'},
        'y': [
            {'key': 'rew', 'label': 'EpRewMean'},
            {'key': 'loss_ent', 'label': 'loss_ent'},
            {'key': 'loss_kl', 'label': 'loss_kl'},
            {'key': 'loss_pol_surr', 'label': 'loss_pol_surr'},
            {'key': 'loss_vf_loss', 'label': 'loss_vf_loss'}
        ],
        'meta': {}
    }
    r = requests.post('https://minotaur-1.herokuapp.com/api/contexts', headers=headers, json=context_spec)
    print('created context: "{0}"'.format(r.json()['name']))
    return r.json()['id']


def train(num_timesteps, seed, model_path=None):
    env_id = 'Odie-v2'
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)
    env = Monitor(env, './video')

    minotaur_context_id = create_minotaur_context()
    def callback(locals, globals):
        try:
            if (len(locals['rewbuffer'])):
                i = locals['iters_so_far']
                meanlosses = locals['meanlosses']
                loss_names = locals['loss_names']
                data = {
                    'i': locals['iters_so_far'],
                    'rew': np.mean(locals['rewbuffer']),
                }
                for (lossval, name) in zipsame(locals['meanlosses'], locals['loss_names']):
                    if (name != 'pol_entpen'):
                        data['loss_' + name] = float(lossval)
                headers = {'Authorization': 'Bearer {0}'.format(jwt)}
                r = requests.patch(
                    'https://minotaur-1.herokuapp.com/api/context/{0}/data'.format(minotaur_context_id),
                    headers=headers,
                    json=data)
                if r.status_code != 201:
                    print(r.json())
        except:
            print('[ERR]', sys.exc_info())

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    # env = RewScale(env, 0.1)
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
            callback=callback
        )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale

def main():
    logger.configure()
    parser = mujoco_arg_parser()
    parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'humanoid_policy'))
    parser.set_defaults(num_timesteps=int(2e7))

    args = parser.parse_args()
    if not args.play:
        # train the model
        train(num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(num_timesteps=1, seed=args.seed)
        U.load_state(args.model_path)
        env = make_mujoco_env('Humanoid-v2', seed=0)

        ob = env.reset()
        while True:
            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, _ =  env.step(action)
            env.render()
            if done:
                ob = env.reset()




if __name__ == '__main__':
    main()
