from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import optax

PRNGKey = jnp.ndarray


class Learner:
    def __init__(
            self,
            model_fn: Callable,
            seed: PRNGKey,
            optimizer_config: dict,
            deterministic: bool,
            *input_example
    ):
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config['clip']),
            optax.scale_by_adam(eps=optimizer_config['eps']),
            optax.scale(-optimizer_config['lr']))
        self.model = hk.transform(model_fn)
        if deterministic:
            self.model = hk.without_apply_rng(self.model)
        self.params = self.model.init(seed, *input_example)
        self.opt_state = self.optimizer.init(self.params)

    @property
    def apply(self):
        return self.model.apply


def td_error(next_value, reward, terminal, discount):
    td = reward + (1.0 - terminal) * discount * next_value
    return td


@jax.jit
def clone_model(a_params, b_params, tau=1.0):
    return jax.tree_multimap(lambda p_a, p_b: p_a * tau + p_b * (1.0 - tau),
                             a_params, b_params)
