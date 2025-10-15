import equinox as eqx   
import jax.numpy as jnp
import jax
import optax
import numpy as np
class GradModel(eqx.Module):
    layers: list

    def __init__(self, in_dim, out_dim, width, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        for i in range(depth):
            self.layers.append(eqx.nn.Linear(in_dim if i == 0 else width, width, key=keys[i]))
        self.layers.append(eqx.nn.Linear(width, out_dim, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)
