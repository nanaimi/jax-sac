from functools import partial
from typing import Callable, Tuple, Union, List

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

PRNGKey = jnp.ndarray


class Learner:
    def __init__(
            self,
            model: hk.Module,
            input_shape: Union[Tuple[int, ...], List[int]],
            seed: PRNGKey,
            optimizer_config: dict
    ):
        self.optimizer = optax.adam(optimizer_config['lr'])
        self.model = hk.transform(model)
        self.params = self.model.init(
            seed,
            np.zeros((1, *input_shape), jnp.float32)
        )
        self.opt_state = self.optimizer.init(self.params)

    @property
    def apply(self):
        return self.model.apply


class LearnableModel:
    def __init__(
            self,
            model: Union[Callable, hk.Module],
            input_shape: Union[Tuple[int, ...], List[int]],
            optimizer: optax.GradientTransformation
    ):
        super().__init__()
        self.model = hk.transform(model)
        self.optimizer = optimizer
        self.params = self.model.init(jax.random.PRNGKey(42), np.zeros(input_shape))
        self.opt_state = self.optimizer.init(self.params)

    def __call__(self, x):
        return self.model.apply(self.params, x=x, rng=jax.random.PRNGKey(42))

    @property
    def apply(self):
        return self.model.apply

    def update(
            self,
            loss_fn: Callable,
            *x
    ) -> dict:
        self.params, self.opt_state, info = self._apply_gradients(
            loss_fn,
            self.opt_state,
            self.params,
            *x
        )
        return info

    @partial(jax.jit, static_argnums=(0, 1))
    def _apply_gradients(self, loss, state, params, *x):
        grad, info = jax.grad(loss, has_aux=True)(params, *x)
        updates, new_state = self.optimizer.update(grad, state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, info


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
