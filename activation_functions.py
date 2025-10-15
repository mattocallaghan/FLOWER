import math
from typing import ClassVar

import jax.numpy as jnp
from flowjax.bijections.bijection import AbstractBijection
import jax


_ACTIVATIONS = {}
def register_activation(cls=None, *, name=None):
  """A decorator for registering activation function bijections."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _ACTIVATIONS:
      raise ValueError(f'Already registered bijection with name: {local_name}')
    _ACTIVATIONS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)

def get_activation_from_name(name):
  return _ACTIVATIONS[name]


def get_activation(config):
  """Create and return the bijection flow.

  """
  activation = config.model.activation

  return get_activation_from_name(activation)()#### note the bracket


@register_activation(name='arctan')
class arctan(AbstractBijection):
    r"""arctan bijection :math:`y=\arctan(x)`."""
    
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    name: ClassVar[str] = "arctan"

    def transform(self, x, condition=None):
        return jnp.arctan(x)

    def transform_and_log_det(self, x, condition=None):
        return jnp.arctan(x), jnp.sum(-1*jnp.log((1.0+x**2)))

    def inverse(self, y, condition=None):
        return jnp.tan(y)

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.tan(y)
        return x, -jnp.sum(-1*jnp.log((1.0+x**2)))


@register_activation(name='gelu')
class arctan(AbstractBijection):
    r"""arctan bijection :math:`y=\arctan(x)`."""
    
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    name: ClassVar[str] = "arctan"

    def transform(self, x, condition=None):
        return jnp.arctan(x)

    def transform_and_log_det(self, x, condition=None):
        return jnp.arctan(x), jnp.sum(-1*jnp.log((1.0+x**2)))

    def inverse(self, y, condition=None):
        return jnp.tan(y)

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.tan(y)
        return x, -jnp.sum(-1*jnp.log((1.0+x**2)))



@register_activation(name='piecewise_arctan')
class piecewise_arctan(AbstractBijection):
    r"""piecewise_arctan bijection :math:`y=\arctan(x)`."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    name: ClassVar[str] = "piecewise_arctan"
    def transform(self, x, condition=None):
        cutoff = jnp.abs(x) >= 500.0
        ones=jnp.ones_like(x)*jnp.sign(x)*3.14159265358979323846/2
        tan_inv = jnp.arctan(x)
        return jnp.where(cutoff, ones, tan_inv)

    def transform_and_log_det(self, x, condition=None):
        cutoff = jnp.abs(x) >= 700.0
        tan_inv = jnp.arctan(x)
        ones=jnp.ones_like(x)*jnp.sign(x)*3.14159265358979323846/2

        jacobian=jnp.sum(jnp.log(1.0/(1.0+x**2)))
        jacobian_flat=-40.0
        return jnp.where(cutoff, ones, tan_inv), jnp.where(cutoff, jacobian_flat, jacobian)

    def inverse(self, y, condition=None):
        return jnp.tan(y)

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.tan(y)
        return x, -jnp.sum(jnp.log(1.0/(1.0+x**2)))
    
@register_activation(name='root_exp')
class root_exp(AbstractBijection):
    r"""arctan bijection :math:`y=\root_exp(x)`."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    name: ClassVar[str] = "root_exp"
    def transform(self, x, condition=None):
        return jnp.sqrt(jnp.exp(x))

    def transform_and_log_det(self, x, condition=None):
        return jnp.sqrt(jnp.exp(x)), jnp.sum(jnp.log((0.5)*jnp.sqrt(jnp.exp(x))))

    def inverse(self, y, condition=None):
        return jnp.log(y**2)

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.log(y**2)
        return x, -jnp.sum(jnp.log((0.5)*jnp.sqrt(jnp.exp(x))))
    
@register_activation(name='softplus')
class softplus(AbstractBijection):
    r"""arctan bijection :math:`y=\arctan(x)`."""
    
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    name: ClassVar[str] = "softlplus"

    def transform(self, x, condition=None):
        return jax.nn.softplus(x)

    def transform_and_log_det(self, x, condition=None):
        return jax.nn.softplus(x), jnp.sum(-jax.nn.softplus(-x))

    def inverse(self, y, condition=None):
        return jnp.log(-1.0+jnp.exp(y))

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.log(-1.0+jnp.exp(y))
        return x, -jnp.sum(-jax.nn.softplus(-x))
    


def _sigmoid_log_grad(x):
    # Log gradient vector of tanh transformation.
    return -1.0 * (x + 2.0*jax.nn.softplus(-x))
@register_activation(name='sigmoid')
class sigmoid(AbstractBijection):
    r"""arctan bijection :math:`y=\arctan(x)`."""
    
    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    name: ClassVar[str] = "sigmoid"

    def transform(self, x, condition=None):
        return jax.nn.sigmoid(x)

    def transform_and_log_det(self, x, condition=None):
        y=jax.nn.sigmoid(x)
        return y, jnp.sum(_sigmoid_log_grad(x))

    def inverse(self, y, condition=None):
        return jnp.log((y-1)/y)

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.log((y-1)/y)
        return x, -jnp.sum(_sigmoid_log_grad(x))
    
#taken directly from flowjax
def _tanh_log_grad(x):
    # Log gradient vector of tanh transformation.
    return -2 * (x + jax.nn.softplus(-2 * x) - jnp.log(2.0))
@register_activation(name='tanh')
class Tanh(AbstractBijection):
    r"""Tanh bijection :math:`y=\tanh(x)`."""

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None
    name: ClassVar[str] = "tanh"
    def transform(self, x, condition=None):
        return jnp.tanh(x)

    def transform_and_log_det(self, x, condition=None):
        return jnp.tanh(x), jnp.sum(_tanh_log_grad(x))

    def inverse(self, y, condition=None):
        return jnp.arctanh(y)

    def inverse_and_log_det(self, y, condition=None):
        x = jnp.arctanh(y)
        return x, -jnp.sum(_tanh_log_grad(x))