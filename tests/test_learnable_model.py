from functools import partial
from typing import Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

from sac.utils import LearnableModel

Batch = Mapping[str, np.ndarray]


def load_dataset(split: str,
                 *,
                 is_training: bool,
                 batch_size: int):
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


@jax.jit
def accuracy(batch):
    predictions = model(batch['image'])
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])


class DummyClassification(object):
    def __init__(self, input_shape):
        def forward(x):
            x = x.astype(jnp.float32) / 255.
            mlp = hk.Sequential([
                hk.Flatten(),
                hk.Linear(300), jax.nn.relu,
                hk.Linear(100), jax.nn.relu,
                hk.Linear(10),
            ])
            return mlp(x)

        self.model = LearnableModel(
            forward,
            input_shape,
            optax.adam(1e-3)
        )

    @partial(jax.jit, static_argnums=0)
    def __call__(self, x):
        return self.model(x)

    def update(self, batch: Batch):
        def loss_fn(params, batch):
            images = batch['image']
            logits = self.model.apply(params, jax.random.PRNGKey(42), images)
            labels = batch["label"]
            loss = jnp.mean(softmax_cross_entropy(logits, labels))
            return loss, {'loss': loss}

        return self.model.update(loss_fn, batch)


if __name__ == '__main__':
    train = load_dataset("train", is_training=True, batch_size=4)
    train_eval = load_dataset("train", is_training=False, batch_size=4)
    test_eval = load_dataset("test", is_training=False, batch_size=4)
    model = DummyClassification(next(train)['image'].shape)
    best_accuracy = 0.0
    for step in range(10001):
        if step % 10 == 0:
            train_accuracy = accuracy(model, next(train_eval))
            test_accuracy = accuracy(model, next(test_eval))
            train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
            print(f"[Step {step}] Train / Test accuracy: "
                  f"{train_accuracy:.3f} / {test_accuracy:.3f}.")
            best_accuracy = max(best_accuracy, test_accuracy)
        model.update(next(train))
