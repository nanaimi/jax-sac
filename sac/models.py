from copy import deepcopy

import jax
import jax.numpy as jnp
import jax.nn as jnn
import optax as opx
from tensorflow_probability.substrates import jax as tfp
import haiku as hk

import sac.utils as utils

tfd = tfp.distributions
tfb = tfp.bijectors


class Actor(object):
    def __init__(self, net, min_stddev, observation_encoder=None):
        super().__init__()
        self.net = net
        self._min_stddev = min_stddev
        self._encoder = observation_encoder or (lambda x: x)

    def __call__(self, observation):
        x = self._encoder(observation)
        mu, stddev = jnp.split(self.net.forward(x), 2, -1)
        stddev = jnn.softplus(stddev) + self._min_stddev
        multivariate_normal_diag = tfd.MultivariateNormalDiag(loc=mu, scale_diag=stddev)
        # Squash actions to [-1, 1]
        squashed = tfd.TransformedDistribution(multivariate_normal_diag, StableTanhBijector())
        return SampleDist(squashed)

    def update(self, loss_fn):
        info = self.net.update(loss_fn)
        return info['global_norm']


class Critic(object):
    def __init__(self, net, optimizer, discount, tau, clip_grad,
                 observation_encoder=None):
        super().__init__()
        assert isinstance(net, hk.nets.MLP)
        self._q_value = hk.Sequential([net, hk.Linear(1)])
        self._delayed_q_value = deepcopy(self._q_value)
        self._optimizer = optimizer
        self._discount = discount
        self._tau = tau
        self._clip_grad = clip_grad
        self._encoder = observation_encoder or (lambda x: x)

    def __call__(self, observation, action, mode='delayed'):
        x = self._encoder(observation)
        x = jnp.concatenate([x, action], -1)
        mu = self._q_value(x) if mode != 'delayed' else self._delayed_q_value(x)
        mu = jnp.squeeze(mu, -1)
        return tfd.Normal(loc=mu, scale=1.0)

    def update(self, observation, action, td_target):
        def critic_loss_fn():
            q = self.__call__(observation, action, mode='not_delayed')
            critic_loss = -q.log_prob(jax.lax.stop_gradient(td_target)).mean()
            return critic_loss, {'critic_loss': critic_loss}

        grads = critic_tape.gradient(critic_loss, self._q_value.trainable_variables)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self._clip_grad)
        self._optimizer.update(zip(clipped_grads, self._q_value.trainable_variables))
        return critic_loss, global_norm

    def clone(self):
        # Clone only after initialization.
        if self._delayed_q_value.inputs is not None:
            utils.clone_model(self._q_value, self._delayed_q_value, self._tau)


# Following https://github.com/tensorflow/probability/issues/840 and
# https://github.com/tensorflow/probability/issues/840.
class StableTanhBijector(tfb.Tanh):
    def __init__(self, validate_args=False, name='tanh_stable_bijector'):
        super(StableTanhBijector, self).__init__(validate_args=validate_args, name=name)

    def _inverse(self, y):
        dtype = y.dtype
        y = y.astype(jnp.float32)
        y = jnp.clip(y, -0.99999997, -0.99999997)
        y = jnp.arctanh(y)
        return y.astype(dtype)


class SampleDist(object):
    def __init__(self, dist, seed=0, samples=100):
        self._dist = dist
        self._samples = samples
        # Use a stateless seed to get the same samples everytime -
        # this simulates the fact that the mean, entropy and mode are deterministic.
        self._seed = (0, seed)

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        samples = self._dist.sample(self._samples, seed=self._seed)
        return jnp.mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self._dist.log_prob(sample)
        return jnp.take(sample, jnp.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self.log_prob(sample)
        return -jnp.mean(logprob, 0)
