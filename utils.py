
from typing import TypeVar
import flax
import jax
import jax.numpy as jnp
import tensorflow as tf
import optax
import numpy as np
#import sbibm


T = TypeVar("T")


def batch_add(a, b):
  return jax.vmap(lambda a, b: a + b)(a, b)


def batch_mul(a, b):
  return jax.vmap(lambda a, b: a * b)(a, b)


def load_training_state(filepath, state):
  with tf.io.gfile.GFile(filepath, "rb") as f:
    state = flax.serialization.from_bytes(state, f.read())
  return state





def flatten_dict(config):
  """Flatten a hierarchical dict to a simple dict."""
  new_dict = {}
  for key, value in config.items():
    if isinstance(value, dict):
      sub_dict = flatten_dict(value)
      for subkey, subvalue in sub_dict.items():
        new_dict[key + "/" + subkey] = subvalue
    elif isinstance(value, tuple):
      new_dict[key] = str(value)
    else:
      new_dict[key] = value
  return new_dict



def get_optimizer(config):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer.lower() == "adam":
        if hasattr(config.optim, "linear_decay_steps"):  # for progressive distillation
            stable_training_schedule = optax.linear_schedule(
                init_value=config.optim.lr,
                end_value=0.0,
                transition_steps=config.optim.linear_decay_steps,
            )
        else:
            stable_training_schedule = optax.constant_schedule(config.optim.lr)
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=0,
                    end_value=config.optim.lr,
                    transition_steps=config.optim.warmup,
                ),
                stable_training_schedule,
            ],
            [config.optim.warmup],
        )

        if not jnp.isinf(config.optim.grad_clip):
            optimizer = optax.chain(
                optax.clip_by_global_norm(max_norm=config.optim.grad_clip),
                optax.adamw(
                    learning_rate=schedule,
                    b1=config.optim.beta1,
                    eps=config.optim.eps,
                    weight_decay=config.optim.weight_decay,
                ),
            )
        else:
            optimizer = optax.adamw(
                learning_rate=schedule,
                b1=config.optim.beta1,
                eps=config.optim.eps,
                weight_decay=config.optim.weight_decay,
            )

    elif config.optim.optimizer.lower() == "radam":
        beta1 = config.optim.beta1
        beta2 = config.optim.beta2
        eps = config.optim.eps
        weight_decay = config.optim.weight_decay
        lr = config.optim.lr
        optimizer = optax.chain(
            optax.scale_by_radam(b1=beta1, b2=beta2, eps=eps),
            optax.add_decayed_weights(weight_decay, None),
            optax.scale(-lr),
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )


    return optimizer





