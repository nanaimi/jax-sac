import haiku as hk
import jax
import optax


class Optimizer(object):
    def __init__(self, grads_transform):
        self.grads_transform = grads_transform
        self._state = None

    def apply_gradients(self, params, grads):
        self._state = self._state or self.grads_transform.init(params)
        updates, self._state = self.grads_transform.update(grads, self._state)
        return optax.apply_updates(params, updates)


class LearnableModel(object):
    def __init__(self, output_sizes, optimizer):
        super().__init__()
        self._net = hk.transform(lambda x: hk.nets.MLP(output_sizes)(x))
        self.params = None
        self.optimizer = optimizer

    @property
    def forward(self):
        def _forward(inputs):
            self.params = self.params or self._net.init(hk.next_rng_key(), inputs)
            return self._net.apply(self.params, inputs, rng=hk.next_rng_key())
        return _forward

    @jax.jit
    def update(self, loss_fn):
        assert self.params
        grads = jax.grad(loss_fn)(self.params)
        self.params, info = self.optimizer.apply_gradients(self.params, grads)
        return info


def td_error(next_value, reward, terminal, discount):
    td = reward + (1.0 - terminal) * discount * next_value
    return td


def normalize_clip(data, mean, stddev, clip):
    stddev = tf.where(stddev < 1e-6, tf.cast(1.0, stddev.dtype), stddev)
    return tf.clip_by_value((data - mean) / stddev,
                            -clip, clip)


def clone_model(a, b, tau=1.0):
    for var_a, var_b in zip(a.trainable_variables, b.trainable_variables):
        var_b.assign(tau * var_a + (1.0 - tau) * var_b)
