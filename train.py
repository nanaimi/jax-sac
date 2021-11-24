from copy import deepcopy

import gym
import numpy as np
import tensorflow as tf
from portfolio_environment.make import create_portfolio_environment
from tensorflow import keras as tfk

import sac.env_wrappers as env_wrappers
import sac.models as models
import train_utils as train_utils
from sac.logger import TrainingLogger
from sac.replay_buffer import ReplayBuffer
from sac.sac import SAC


def make_net_optimizer_from_config(net_dict, optimizer_dict):
    net = tfk.Sequential()
    for _ in range(net_dict['layers']):
        net.add(tfk.layers.Dense(net_dict['units'], net_dict['act']))
    optimizer = tfk.optimizers.Adam(optimizer_dict['lr'],
                                    epsilon=optimizer_dict['eps'])
    return net, optimizer


def make_agent(config, environment, logger):
    encoder = None if config.task != 'PortfolioEnv' else models.PortfolioObservationEncoder(32, 20)
    actor = models.Actor(*make_net_optimizer_from_config(config.actor, config.actor_opt),
                         environment.action_space.shape[0],
                         config.actor['min_std'],
                         config.actor_opt['clip'],
                         encoder)
    critics = [models.Critic(*make_net_optimizer_from_config(config.critic, config.critic_opt),
                             config.discount,
                             config.tau,
                             config.critic_opt['clip'],
                             deepcopy(encoder))
               for _ in range(config.critics)]
    experience = ReplayBuffer(config.replay['capacity'], config.seed, config.replay['batch'])
    agent = SAC(actor, critics, experience, logger, environment.action_space, config)
    return agent


def make_environment(config):
    if config.task == 'PortfolioEnv':
        env = create_portfolio_environment(config.time_limit, config.window_length,
                                           config.seed, fees_rate=config.fees_rate)
    else:
        env = gym.make(config.task)
    if not isinstance(env, gym.wrappers.TimeLimit):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=config.time_limit)
    else:
        # https://github.com/openai/gym/issues/499
        env._max_episode_steps = config.time_limit
    env = env_wrappers.ActionRepeat(env, config.action_repeat)
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    env = gym.wrappers.TransformReward(
        env,
        dict(tanh=np.tanh, none=lambda x: x)[config.clip_rewards]
    )
    env.seed(config.seed)
    return env


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    config = train_utils.load_config()
    environment = make_environment(config)
    logger = TrainingLogger(config.log_dir)
    agent = make_agent(config, environment, logger)
    train_utils.train(config, agent, environment, logger)
