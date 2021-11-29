import functools
import os
import pickle
from copy import deepcopy
from typing import Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import sac.utils as utils
from sac.models import EntropyBonus

PRNGKey = jnp.ndarray
Observation = np.ndarray
Batch = Mapping[str, np.ndarray]


class SAC:
    def __init__(
            self,
            observation_space,
            action_space,
            actor,
            critics,
            experience,
            logger,
            config
    ):
        super(SAC, self).__init__()
        self.rng_seq = hk.PRNGSequence(config.seed)
        self.actor = utils.Learner(actor, next(self.rng_seq), config.actor_opt,
                                   True, observation_space.sample())
        self.critics = utils.Learner(critics,
                                     next(self.rng_seq),
                                     config.critic_opt, True,
                                     observation_space.sample(),
                                     action_space.sample())
        self.target_critics = deepcopy(self.critics)
        self.entropy_bonus = utils.Learner(
            lambda log_pi: EntropyBonus()(log_pi), next(self.rng_seq),
            config.alpha_opt, True, np.ones((1,)))
        self.experience = experience
        self.logger = logger
        self.config = config
        self.training_step = 0
        self._target_entropy = -np.prod(action_space.shape)
        self._prefill_policy = lambda: np.random.uniform(action_space.low,
                                                         action_space.high)

    def __call__(self, observation: Observation, training=True):
        if self.training_step <= self.config.prefill and training:
            return self._prefill_policy()
        if self.time_to_update and training:
            for batch in self.experience.sample(self.config.train_steps):
                self.update_actor_critic(batch)
        if self.time_to_log and training:
            self.logger.log_metrics(self.training_step)
        if self.time_to_clone_critics:
            self.target_critics.params = utils.clone_model(
                self.critics.params, self.target_critics.params)

        action = self.policy(observation, self.actor.params,
                             next(self.rng_seq), training)
        return np.clip(action, -1.0, 1.0)

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def policy(self, observation: Observation, params: hk.Params,
               rng_key: PRNGKey, training=True) -> jnp.ndarray:
        policy = self.actor.apply(params, observation)
        action = policy.sample(seed=rng_key) if training else policy.mode(
            seed=rng_key)
        return action

    def observe(self, transition):
        self.training_step += 1
        self.experience.store(**transition)

    def update_actor_critic(self, batch):
        self.critics.params, self.critics.opt_state, critic_report = \
            self._update_critics(self.actor.params, self.critics.params,
                                 self.target_critics.params,
                                 self.entropy_bonus.params,
                                 next(self.rng_seq), self.critics.opt_state,
                                 batch)
        self.actor.params, self.actor.opt_state, actor_report = \
            self._update_actor(self.actor.params, self.critics.params,
                               self.entropy_bonus.params, next(self.rng_seq),
                               self.actor.opt_state, batch)
        (
            self.entropy_bonus.params,
            self.entropy_bonus.opt_state,
            entropy_report
        ) = self._update_alpha(self.actor.params,
                               self.entropy_bonus.params,
                               next(self.rng_seq),
                               self.entropy_bonus.opt_state,
                               batch)
        report = {**critic_report, **actor_report, **entropy_report}
        for k, v in report.items():
            self.logger[k].update_state(v)

    @functools.partial(jax.jit, static_argnums=0)
    def _update_critics(
            self,
            actor_params: hk.Params,
            critic_params: hk.Params,
            target_critic_params: hk.Params,
            entropy_params: hk.Params,
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            batch: Batch
    ) -> Tuple[hk.Params, optax.OptState, dict]:
        def loss(critic_params: hk.Params, target_critic_params: hk.Params,
                 actor_params: hk.Params, entropy_params: hk.Params,
                 rng_key: PRNGKey):
            policy = self.actor.apply(actor_params, batch['next_observation'])
            next_action = policy.sample(seed=rng_key)
            next_qs = self.target_critics.apply(target_critic_params,
                                                batch['next_observation'],
                                                next_action)
            debiased_q = jnp.min(
                jnp.array(list(map(lambda q: q.mean(), next_qs))), 0)
            entropy_bonus = self.entropy_bonus.apply(
                entropy_params, policy.log_prob(next_action))
            soft_q = debiased_q + entropy_bonus
            soft_q_target = utils.td_error(
                soft_q, batch['reward'] * self.config.reward_scale,
                batch['terminal'], self.config.discount)
            qs = self.critics.apply(critic_params, batch['observation'],
                                    batch['action'])
            critics_loss = sum((map(lambda q: -jnp.mean(q.log_prob(
                jax.lax.stop_gradient(soft_q_target))), qs)))
            return critics_loss, {'agent/critic/loss': critics_loss}

        grads, report = jax.grad(loss, has_aux=True)(critic_params,
                                                     target_critic_params,
                                                     actor_params,
                                                     entropy_params, rng_key)
        updates, new_opt_state = self.critics.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(critic_params, updates)
        report.update({'agent/critic/grads': optax.global_norm(grads)})
        return new_params, new_opt_state, report

    @functools.partial(jax.jit, static_argnums=0)
    def _update_actor(
            self,
            actor_params: hk.Params,
            critic_params: hk.Params,
            entropy_params: hk.Params,
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            batch: Batch
    ) -> Tuple[hk.Params, optax.OptState, dict]:
        observation = batch['observation']

        def loss(actor_params: hk.Params, critic_params: hk.Params,
                 entropy_params: hk.Params, rng_key: PRNGKey):
            policy = self.actor.apply(actor_params, observation)
            action = policy.sample(seed=rng_key)
            qs = self.critics.apply(critic_params, observation, action)
            debiased_q = jnp.min(
                jnp.array(list(map(lambda q: q.mean(), qs))), 0)
            entropy_bonus = self.entropy_bonus.apply(entropy_params,
                                                     policy.log_prob(action))
            actor_loss = -jnp.mean(debiased_q + entropy_bonus)
            return actor_loss, {
                'agent/actor/loss': actor_loss}

        grads, report = jax.grad(loss, has_aux=True)(actor_params,
                                                     critic_params,
                                                     entropy_params, rng_key)
        updates, new_opt_state = self.actor.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(actor_params, updates)
        report.update({'agent/actor/grads': optax.global_norm(grads)})
        return new_params, new_opt_state, report

    @functools.partial(jax.jit, static_argnums=0)
    def _update_alpha(
            self,
            actor_params: hk.Params,
            entropy_params: hk.Params,
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            batch: Batch
    ) -> Tuple[jnp.ndarray, optax.OptState, dict]:
        policy = self.actor.apply(actor_params, batch['observation'])
        action = policy.sample(seed=rng_key)
        log_pi = policy.log_prob(action)

        def loss(params: hk.Params):
            entropy_bonus = self.entropy_bonus.apply(params, log_pi)
            alpha_loss = jnp.mean(-entropy_bonus + self._target_entropy)
            return alpha_loss, {'actor/alpha/loss': alpha_loss}

        grads, report = jax.grad(loss, has_aux=True)(entropy_params)
        updates, new_opt_state = self.entropy_bonus.optimizer.update(
            grads, opt_state)
        new_params = optax.apply_updates(entropy_params, updates)
        report.update({'agent/alpha/grads': optax.global_norm(grads),
                       'agent/actor/entropy': -jnp.mean(log_pi)})
        return new_params, new_opt_state, report

    def write(self, path):
        with open(os.path.join(path, 'checkpoint.pickle'), 'wb') as f:
            pickle.dump({'actor': self.actor,
                         'critics': self.critics,
                         'entropy': self.entropy_bonus,
                         'experience': self.experience,
                         'training_steps': self.training_step}, f)

    def load(self, path):
        with open(os.path.join(path, 'checkpoint.pickle'), 'rb') as f:
            data = pickle.load(f)
        for key, obj in zip(data.keys(), [
            self.actor,
            self.critics,
            self.entropy_bonus,
            self.experience,
            self.training_step
        ]):
            obj = data[key]

    @property
    def time_to_update(self):
        return self.training_step > self.config.prefill and \
               self.training_step % self.config.train_every == 0

    @property
    def time_to_clone_critics(self):
        return self.training_step > self.config.prefill and \
               self.training_step % self.config.slow_target_update == 0

    @property
    def time_to_log(self):
        return self.training_step and self.training_step % \
               self.config.log_every == 0
