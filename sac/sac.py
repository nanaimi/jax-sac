import jax.numpy as jnp
import numpy as np
import haiku as hk

import sac.utils as utils


class SAC(hk.Module):
    def __init__(self, actor, critics, experience,
                 logger, action_space, config):
        super(SAC, self).__init__()
        self.actor = actor
        self.critics = critics
        self.experience = experience
        self.logger = logger
        self.config = config
        self.training_step = 0
        self._log_alpha = tf.Variable(0.0)
        self._alpha_optimizer = tf.optimizers.Adam(config.alpha_lr)
        self._target_entropy = -np.prod(action_space.shape)
        self._prefill_policy = lambda: np.random.uniform(action_space.low, action_space.high)

    def __call__(self, observation, training=True):
        if self.training_step <= self.config.prefill and training:
            return self._prefill_policy()
        if self.time_to_update and training:
            for batch in self.experience.sample(self.config.train_steps):
                self.update_actor_critic(batch)
        if self.time_to_log and training:
            self.logger.log_metrics(self.training_step)
        if self.time_to_clone_critics:
            for critic in self.critics:
                critic.clone()

        def standardize(observation):
            if isinstance(observation, np.ndarray):
                observation = tf.constant(observation, tf.float32)[None]
            else:
                observation = tf.nest.map_structure(lambda x: tf.constant(x, tf.float32)[None],
                                                    observation)
            return observation

        action = self.policy(standardize(observation), training).numpy()
        return np.clip(action, -1.0, 1.0)

    @tf.function
    def policy(self, observation, training=True):
        policy = self.actor(observation)
        action = policy.sample() if training else policy.mode()
        action = tf.squeeze(action, 0)
        return action

    def observe(self, transition):
        self.training_step.assign_add(1)
        self.experience.store(**transition)

    @tf.function
    def update_actor_critic(self, batch):
        report = dict()
        critic_report = self._update_critics(batch)
        report.update(critic_report)
        actor_report = self._update_actor(batch)
        report.update(actor_report)
        alpha_report = self._update_alpha(batch)
        report.update(alpha_report)
        report.update({'agent/actor/entropy': self.actor(batch['observation']).entropy()})
        for k, v in report.items():
            self.logger[k].update_state(v)

    def _update_critics(self, batch):
        policy = self.actor(batch['next_observation'])
        next_action = policy.sample()
        next_qs = [critic(batch['next_observation'], next_action, mode='delayed').mean()
                   for critic in self.critics]
        debiased_q = tf.reduce_min(next_qs, axis=0)
        entropy_bonus = -tf.exp(self._log_alpha) * policy.log_prob(next_action)
        soft_q = debiased_q + entropy_bonus
        soft_q_target = utils.td_error(soft_q, batch['reward'] * self.config.reward_scale,
                                       batch['terminal'], self.config.discount)
        report = {}
        for i, critic in enumerate(self.critics):
            loss, grads = critic.update(batch['observation'], batch['action'], soft_q_target)
            report['agent/critic_' + str(i) + '/loss'] = loss
            report['agent/critic_' + str(i) + '/grads'] = grads
        return report

    def _update_actor(self, batch):
        observation = batch['observation']
        with tf.GradientTape() as actor_tape:
            policy = self.actor(observation)
            action = policy.sample()
            q_values = tf.reduce_min([critic(observation, action, mode='not_delayed').mean()
                                      for critic in self.critics], axis=0)
            log_pi = policy.log_prob(action)
            entropy_bonus = -tf.exp(self._log_alpha) * log_pi
            loss = -tf.reduce_mean(q_values + entropy_bonus)
        grads = self.actor.update(loss, actor_tape)
        return {'agent/actor/loss': loss,
                'agent/actor/grads': grads}

    def _update_alpha(self, batch):
        policy = self.actor(batch['observation'])
        action = policy.sample()
        log_pi = policy.log_prob(action)
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(
                tf.exp(self._log_alpha) * tf.stop_gradient(log_pi + self._target_entropy)
            )
        grads = tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(grads, [self._log_alpha]))
        return {'agent/alpha/loss': alpha_loss,
                'agent/alpha/grads': grads}

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
        return self.training_step and self.training_step % self.config.log_every == 0
