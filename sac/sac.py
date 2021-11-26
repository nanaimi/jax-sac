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
Batch = Mapping[str, np.ndarray]


def create_optimizer(learning_rate, clip):
    return optax.adam(learning_rate)


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
        self.actor = utils.Learner(
            actor,
            observation_space.shape,
            next(self.rng_seq),
            config.actor_opt
        )
        self.critics = utils.Learner(
            critics,
            observation_space.shape + action_space.shape,
            next(self.rng_seq),
            config.critic_opt
        )
        self.target_critics = deepcopy(self.critics)
        self.entropy_bonus = utils.Learner(
            EntropyBonus(),
            (1,),
            next(self.rng_seq),
            config.alpha_opt
        )
        self.experience = experience
        self.logger = logger
        self.config = config
        self.training_step = 0
        self._target_entropy = -np.prod(action_space.shape)
        self._prefill_policy = lambda: np.random.uniform(action_space.low,
                                                         action_space.high)

    def __call__(self, observation, training=True):
        if self.training_step <= self.config.prefill and training:
            return self._prefill_policy()
        if self.time_to_update and training:
            for batch in self.experience.sample(self.config.train_steps):
                self.update_actor_critic(batch)
        if self.time_to_log and training:
            self.logger.log_metrics(self.training_step)
        if self.time_to_clone_critics:
            print("cloning!")

        action = self.policy(observation, training).numpy()
        return np.clip(action, -1.0, 1.0)

    @jax.jit
    def policy(self, observation, training=True):
        policy = self.actor.apply(self.actor.params, observation)
        action = policy.sample() if training else policy.mode()
        action = jnp.squeeze(action, 0)
        return action

    def observe(self, transition):
        self.training_step += 1
        self.experience.store(**transition)

    def update_actor_critic(self, batch):
        report = dict()
        critic_report = self._update_critics(batch)
        report.update(critic_report)
        actor_report = self._update_actor(batch)
        report.update(actor_report)
        alpha_report = self._update_alpha(batch)
        report.update(alpha_report)
        report.update(
            {'agent/actor/entropy': self.actor(batch['observation']).entropy()})
        for k, v in report.items():
            self.logger[k].update_state(v)

    @jax.jit
    def _update_critics(
            self,
            params: hk.Params,
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            batch: Batch
    ) -> Tuple[hk.Params, optax.OptState]:
        def loss():
            policy = self.actor.apply(self.actor.params,
                                      batch['next_observation'])
            # TODO (yarden): how can we use down here that rng_key?
            next_action = policy.sample()
            next_qs = self.target_critics.apply(self.target_critics.params,
                                                batch['next_observation'],
                                                next_action)
            debiased_q = jnp.minimum(*jax.tree_map(lambda q: q.mean(), next_qs))
            entropy_bonus = self.entropy_bonus.apply(
                self.entropy_bonus.params, policy.log_prob(next_action)
            )
            soft_q = debiased_q + entropy_bonus
            soft_q_target = utils.td_error(
                soft_q, batch['reward'] * self.config.reward_scale,
                batch['terminal'], self.config.discount)
            qs = self.critics.apply(params, batch['observation'],
                                    batch['action'])
            critics_loss = jnp.sum(jax.tree_map(
                lambda q: -jnp.mean(q.log_prob(
                    jax.lax.stop_gradient(soft_q_target))), qs))
            return critics_loss

        grads = jax.grad(loss)(params)
        updates, new_opt_state = self.critics.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        report = {}
        return new_params, new_opt_state

    def _update_actor(
            self,
            params: hk.Params,
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            batch: Batch
    ) -> Tuple[hk.Params, optax.OptState]:
        observation = batch['observation']

        def loss():
            policy = self.actor.apply(params, observation)
            action = policy.sample()
            qs = self.critics.apply(self.critics.params, observation, action)
            debiased_q = jnp.minimum(*jax.tree_map(lambda q: q.mean(), qs))
            entropy_bonus = self.entropy_bonus.apply(self.entropy_bonus.params,
                                                     policy.log_prob(action))
            return -jnp.mean(debiased_q + entropy_bonus)

        grads = jax.grad(loss)
        updates, new_opt_state = self.actor.optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        report = {}
        return new_params, new_opt_state

    def _update_alpha(
            self,
            params: hk.Params,
            rng_key: PRNGKey,
            opt_state: optax.OptState,
            batch: Batch
    ) -> Tuple[jnp.ndarray, optax.OptState]:
        policy = self.actor.apply(self.actor.params, batch['observation'])
        action = policy.sample()
        log_pi = policy.log_prob(action)

        def loss():
            entropy_bonus = self.entropy_bonus.apply(params, log_pi)
            return jnp.mean(-entropy_bonus + self._target_entropy)

        grads = jax.grad(loss)
        updates, new_opt_state = self.entropy_bonus.optimizer.update(
            grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

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
