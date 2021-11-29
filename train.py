import gym
import haiku.nets as nets
import numpy as np

import sac.env_wrappers as env_wrappers
import sac.models as models
import train_utils as train_utils
from sac.logger import TrainingLogger
from sac.replay_buffer import ReplayBuffer
from sac.sac import SAC


def make_agent(config, environment, logger):
    actor = models.Actor(
        nets.MLP((config.actor['units']) * config.actor['layers'] + (
            environment.action_space.shape[0],)),
        config.actor['min_std']
    )
    critics = models.DoubleCritic(
        *[nets.MLP(
            (config.critic['units']) * config.critic['layers'] + (1,))
            for _ in range(config.critics)]
    )
    experience = ReplayBuffer(config.replay['capacity'], config.seed,
                              config.replay['batch'])
    agent = SAC(environment.observation_space,
                environment.action_space,
                actor, critics, experience,
                logger, config)
    return agent


def make_environment(config):
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
    config = train_utils.load_config()
    environment = make_environment(config)
    logger = TrainingLogger(config.log_dir)
    agent = make_agent(config, environment, logger)
    train_utils.train(config, agent, environment, logger)
