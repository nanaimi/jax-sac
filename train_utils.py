import os.path
import pathlib
from collections import defaultdict

import numpy as np
from tqdm import tqdm


def do_episode(agent, training, environment, config, pbar, render):
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    observation = environment.reset()
    while not done:
        action = agent(observation, training)
        next_observation, reward, done, info = environment.step(action)
        terminal = done and not info.get('TimeLimit.truncated', False)
        if training:
            agent.observe(dict(observation=observation,
                               next_observation=next_observation,
                               action=action.astype(np.float32),
                               reward=np.array(reward, np.float32),
                               terminal=np.array(terminal, np.float32),
                               info=info))
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['action'].append(action)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
        observation = next_observation
        if render:
            episode_summary['image'].append(
                environment.render(mode='rgb_array'))
        pbar.update(config.action_repeat)
        steps += config.action_repeat
    episode_summary['steps'] = [steps]
    return steps, episode_summary


def interact(agent, environment, steps, config, training=True,
             on_episode_end=None):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []
    while steps_count < steps:
        episode_steps, episode_summary = do_episode(agent, training,
                                                    environment, config,
                                                    pbar,
                                                    len(episodes) <
                                                    config.render_episodes and
                                                    not training)
        steps_count += episode_steps
        episodes.append(episode_summary)
        if on_episode_end is not None:
            on_episode_end(episode_summary, steps_count)
    pbar.close()
    return steps, episodes


def make_summary(summaries, prefix):
    epoch_summary = {prefix + '/average_return': np.asarray([
        sum(episode['reward']) for episode in summaries]).mean(),
                     prefix + '/average_episode_length': np.asarray([
                         episode['steps'][0]
                         for episode in summaries]).mean()}
    return epoch_summary


def evaluate(agent, train_env, logger, config, steps):
    evaluation_steps, evaluation_episodes_summaries = interact(
        agent, train_env, config.evaluation_steps_per_epoch, config,
        training=False)
    if config.render_episodes:
        videos = list(map(lambda episode: episode.get('image'),
                          evaluation_episodes_summaries[
                          :config.render_episodes]))
        logger.log_video(
            np.array(videos, copy=False).transpose([0, 1, 4, 2, 3]), steps)
    return make_summary(evaluation_episodes_summaries, 'evaluation')


def on_episode_end(episode_summary, logger, global_step, steps_count):
    episode_return = sum(episode_summary['reward'])
    steps = global_step + steps_count
    print("\nFinished episode with return: {}".format(episode_return))
    summary = {'training/episode_return': episode_return}
    logger.log_evaluation_summary(summary, steps)


def train(config, agent, environment, logger):
    from jax.config import config as jax_config
    jax_config.update('jax_disable_jit', not config.jit)
    np.random.seed(config.seed)
    steps = 0
    if pathlib.Path(config.log_dir, 'agent_data').exists():
        checkpoint.restore(
            os.path.join(config.log_dir, 'agent_data', 'checkpoint'))
        steps = agent.training_step
        print("Loaded {} steps. Continuing training from {}".format(
            steps,
            config.log_dir))
    while steps < config.steps:
        print("Performing a training epoch.")
        training_steps, training_episodes_summaries = interact(
            agent, environment, config.training_steps_per_epoch, config,
            training=True,
            on_episode_end=lambda episode_summary, steps_count: on_episode_end(
                episode_summary, logger=logger, global_step=steps,
                steps_count=steps_count))
        steps += training_steps
        training_summary = make_summary(training_episodes_summaries, 'training')
        if config.evaluation_steps_per_epoch:
            print("Evaluating.")
            evaluation_summaries = evaluate(agent, environment, logger, config,
                                            steps)
            training_summary.update(evaluation_summaries)
        logger.log_evaluation_summary(training_summary, steps)
        checkpoint.write(
            os.path.join(config.log_dir, 'agent_data', 'checkpoint'))
    environment.close()
    return agent


def load_config():
    import argparse
    import ruamel.yaml as yaml

    def args_type(default):
        def parse_string(x):
            if default is None:
                return x
            if isinstance(default, bool):
                return bool(['False', 'True'].index(x))
            if isinstance(default, int):
                return float(x) if ('e' in x or '.' in x) else int(x)
            if isinstance(default, (list, tuple)):
                return tuple(args_type(default[0])(y) for y in x.split(','))
            return type(default)(x)

        def parse_object(x):
            if isinstance(default, (list, tuple)):
                return tuple(x)
            return x

        return lambda x: parse_string(x) if isinstance(x,
                                                       str) else parse_object(x)

    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    with open('sac/config.yaml') as file:
        configs = yaml.safe_load(file)
    defaults = {}
    for name in args.configs:
        defaults.update(configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = args_type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
    return parser.parse_args(remaining)
