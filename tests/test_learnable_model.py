import unittest

import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds

from sac.utils import Optimizer, LearnableModel


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
def accuracy(model, batch):
    x = batch['image'].astype(jnp.float32) / 255.
    predictions = model.forward(x)
    return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])


def loss_fn(model, batch):
    images = batch['images']
    labels = jax.nn.one_hot(batch["label"], 10)
    x = images.astype(jnp.float32) / 255.
    logits = model.forward(x)
    return jnp.mean(softmax_cross_entropy(logits, labels))


class TestModelAndOptimizer(unittest.TestCase):

    def test_it(self):
        model = LearnableModel((300, 100, 10), Optimizer())
        train = load_dataset("train", is_training=True, batch_size=1000)
        train_eval = load_dataset("train", is_training=False, batch_size=10000)
        test_eval = load_dataset("test", is_training=False, batch_size=10000)
        best_accuracy = 0.0
        for step in range(10001):
            if step % 1000 == 0:
                train_accuracy = accuracy(model, next(train_eval))
                test_accuracy = accuracy(model, next(test_eval))
                train_accuracy, test_accuracy = jax.device_get((train_accuracy, test_accuracy))
                print(f"[Step {step}] Train / Test accuracy: "
                      f"{train_accuracy:.3f} / {test_accuracy:.3f}.")
                best_accuracy = max(best_accuracy, test_accuracy)
            model.update(lambda: loss_fn(model, next(train)))
        self.assertLess(best_accuracy, 0.6)
