from functools import partial
from typing import ClassVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from paramax import AbstractUnwrappable, Parameterize
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection
class SmoothBijector(AbstractBijection):
    """Smooth Bijectior. https://arxiv.org/abs/2110.00351


    """
    a: Array
    b: Array
    c: Array
    p: Array
    shape: ClassVar[tuple] = ()
    cond_shape: ClassVar[None] = None
    eps: ClassVar[float] = 1e-3

    def __init__(
        self,
        *,
        a,
        b,
        c,
        p,



    ):
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        
  

    def transform_and_log_det(self, x, condition=None):
        # Following notation from the paper
        return ((self.bijection))([jax.nn.sigmoid(self.a), jax.nn.sigmoid(self.b), jax.nn.sigmoid(self.c), self.p], x),self.logdet_fn(x, self.a, self.b, self.c, self.p)


    def inverse_and_log_det(self, y, condition=None):
        raise NotImplementedError(
            "Inverse methods not implemented for block autoregressive network. "
            "Consider providing a numerical method with ``NumericalInverse``. "
            "See ``flowjax.root_finding`` for numerical methods."
        )
 
    def general_sigmoid(self,x, a, b, c):
      x = jnp.clip(x, self.eps, 1 - self.eps)
      z = (jax.scipy.special.logit(x) +b )* a
      y = jax.nn.sigmoid(z) * (1 - c) + c * x
      return y
    
    def bijection(self,params, x):
      a, b, c, p = params
      a_in, b_in = [0. - 1e-1, 1. + 1e-1]

      x = (x - a_in) / (b_in - a_in)
      x0 = (jnp.zeros_like(x) - a_in)/ ( b_in - a_in)
      x1 = (jnp.ones_like(x) - a_in) /( b_in - a_in)

      y = self.general_sigmoid(x, a, b, c)
      y0 = self.general_sigmoid(x0, a, b, c)
      y1 = self.general_sigmoid(x1, a, b, c)

      y = (y - y0)/(y1 - y0)
      return jnp.sum(p*(y*(1-c) + c *x), axis=0)
    
    def logdet_fn(self,x,a,b,c,p):
        g = jax.grad(self.bijection, argnums=1)([a,b,c,p], x)
        s, logdet = jnp.linalg.slogdet(jnp.atleast_2d(g)+ 1e-8)
        return s*logdet

