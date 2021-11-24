from copy import deepcopy

import tensorflow as tf
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

import sac.utils as utils


class Actor(tf.Module):
    def __init__(self, net, optimizer, size, min_stddev, clip_grad,
                 observation_encoder=None):
        super().__init__()
        assert isinstance(net, tf.keras.Sequential)
        self._net = net
        self._net.add(tf.keras.layers.Dense(2 * size))
        self._optimizer = optimizer
        self._min_stddev = min_stddev
        self._clip_grad = clip_grad
        self._encoder = observation_encoder or (lambda x: x)

    def __call__(self, observation):
        x = self._encoder(observation)
        mu, stddev = tf.split(self._net(x), 2, -1)
        stddev = tf.math.softplus(stddev) + self._min_stddev
        multivariate_normal_diag = tfd.MultivariateNormalDiag(loc=mu, scale_diag=stddev)
        # Squash actions to [-1, 1]
        squashed = tfd.TransformedDistribution(multivariate_normal_diag, StableTanhBijector())
        return SampleDist(squashed)

    def update(self, loss, tape):
        grads = tape.gradient(loss, self._net.trainable_variables)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self._clip_grad)
        self._optimizer.apply_gradients(zip(clipped_grads, self._net.trainable_variables))
        return global_norm


class Critic(tf.Module):
    def __init__(self, net, optimizer, discount, tau, clip_grad,
                 observation_encoder=None):
        super().__init__()
        assert isinstance(net, tf.keras.Sequential)
        self._q_value = net
        self._q_value.add(tf.keras.layers.Dense(1))
        self._delayed_q_value = deepcopy(self._q_value)
        self._optimizer = optimizer
        self._discount = discount
        self._tau = tau
        self._clip_grad = clip_grad
        self._encoder = observation_encoder or (lambda x: x)

    def __call__(self, observation, action, mode='delayed'):
        x = self._encoder(observation)
        x = tf.concat([x, action], -1)
        mu = self._q_value(x) if mode != 'delayed' else self._delayed_q_value(x)
        mu = tf.squeeze(mu, -1)
        return tfd.Normal(loc=mu, scale=1.0)

    def update(self, observation, action, td_target):
        with tf.GradientTape() as critic_tape:
            q = self.__call__(observation, action, mode='not_delayed')
            critic_loss = -tf.reduce_mean(q.log_prob(tf.stop_gradient(td_target)))
        grads = critic_tape.gradient(critic_loss, self._q_value.trainable_variables)
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self._clip_grad)
        self._optimizer.apply_gradients(zip(clipped_grads, self._q_value.trainable_variables))
        return critic_loss, global_norm

    def clone(self):
        # Clone only after initialization.
        if self._delayed_q_value.inputs is not None:
            utils.clone_model(self._q_value, self._delayed_q_value, self._tau)


class PortfolioObservationEncoder(tf.Module):
    def __init__(self, gru_units, units):
        super().__init__()
        self.recurrent = tf.keras.layers.GRU(gru_units, return_sequences=True, return_state=True)
        self.w1 = tf.keras.layers.Dense(units)
        self.w2 = tf.keras.layers.Dense(units)
        self.v = tf.keras.layers.Dense(1)

    def __call__(self, observation):
        state, prev_weights = observation
        outputs, final_state = self.recurrent(state)
        final_state = final_state[:, None]
        score = tf.nn.tanh(self.w1(outputs) + self.w2(final_state))
        attention = tf.nn.softmax(self.v(score), axis=1)
        context = tf.reduce_sum(attention * outputs, axis=1)
        return tf.concat([context, prev_weights], -1)


# Following https://github.com/tensorflow/probability/issues/840 and
# https://github.com/tensorflow/probability/issues/840.
class StableTanhBijector(tfb.Tanh):
    def __init__(self, validate_args=False, name='tanh_stable_bijector'):
        super(StableTanhBijector, self).__init__(validate_args=validate_args, name=name)

    def _inverse(self, y):
        dtype = y.dtype
        y = tf.cast(y, tf.float32)
        y = tf.clip_by_value(y, -0.99999997, -0.99999997)
        y = tf.atanh(y)
        return tf.saturate_cast(y, dtype)


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
        return tf.reduce_mean(samples, 0)

    def mode(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self._dist.log_prob(sample)
        return tf.gather(sample, tf.argmax(logprob))[0]

    def entropy(self):
        sample = self._dist.sample(self._samples, seed=self._seed)
        logprob = self.log_prob(sample)
        return -tf.reduce_mean(logprob, 0)
