"""Block Neural Autoregressive bijection implementation."""

from collections.abc import Callable
from math import prod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import random
from jax.nn import softplus
from jaxtyping import PRNGKeyArray
from paramax import Parameterize, WeightNormalization

from flowjax import masks
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.tanh import _tanh_log_grad
from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.softplus import SoftPlus
from flowjax.bijections.tanh import LeakyTanh, Tanh

class _CallableToBijection(AbstractBijection):
    # Wrap a callable e.g. a function or a callable module. We assume the callable acts
    # on scalar values and log_det can be computed in a stable manner with jax.grad.

    fn: Callable
    shape: ClassVar[tuple] = ()
    cond_shape: ClassVar[None] = None

    def __init__(self, fn: Callable):
        if not callable(fn):
            raise TypeError(f"Expected callable, got {type(fn)}.")
        self.fn = fn

    def transform_and_log_det(self, x, condition=None):
        y, grad = eqx.filter_value_and_grad(self.fn)(x)
        return y, jnp.log(jnp.abs(grad))

    def inverse_and_log_det(self, y, condition=None):
        raise NotImplementedError()




class BNAF_ELU(AbstractBijection):
    r"""Block neural autoregressive flow with ELU input layer and more complex conditional.
    Almost identical to flowjax implementation, but with a different activation function and conditional treatment


    Args:
        key: Jax key
        dim: Dimension of the distribution.
        cond_dim: Dimension of conditioning variables. Defaults to None.
        depth: Number of hidden layers in the network.
        block_dim: Block dimension (hidden layer size is `dim*block_dim`).
        activation: Activation function, either a scalar bijection or a callable that
            computes the activation for a scalar value. Note that the activation should
            be bijective to ensure invertibility of the network and in general should
            map real -> real to ensure that when transforming a distribution (either
            with the forward or inverse), the map is defined across the support of
            the base distribution. Defaults to :math:`\tanh(x) + 0.01 * x`.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    depth: int
    layers: list
    cond_linear: tuple[eqx.nn.Linear,...] | None
    block_dim: int
    activation: AbstractBijection
    activation_tanh: AbstractBijection
    beta: float
    L_x: jnp.ndarray
    U_x: jnp.ndarray
    L_c: jnp.ndarray
    U_c: jnp.ndarray


    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        dim: int,
        cond_dim: int | None = None,
        depth: int,
        block_dim: int,
        activation: AbstractBijection | Callable | None = None,
    ):
        key, subkey = jr.split(key)
        assert activation != None, "activation must be provided."
        assert cond_dim >0, "cond_dim must be provided."
        if isinstance(activation, AbstractBijection):
            if activation.shape != () or activation.cond_shape is not None:
                raise ValueError("Bijection must be unconditional with shape ().")
        else:
            activation = _CallableToBijection(activation)

        layers_and_log_jac_fns = []
        if depth == 0:
            layers_and_log_jac_fns.append(
                block_autoregressive_linear(
                    key,
                    n_blocks=dim,
                    block_shape=(1, 1),
                ),
            )
        else:
            keys = random.split(key, depth + 1)

            block_shapes = [
                (block_dim, 1),
                *[(block_dim, block_dim)] * (depth - 1),
                (1, block_dim),
            ]

            for layer_key, block_shape in zip(keys, block_shapes, strict=True):
                layers_and_log_jac_fns.append(
                    block_autoregressive_linear(
                        layer_key,
                        n_blocks=dim,
                        block_shape=block_shape,
                    ),
                )

        if cond_dim is not None:
            self.cond_linear = []
            # Only create conditioning for the first layer to prevent over-expressiveness
            first_layer_out_dim = layers_and_log_jac_fns[0][0].out_features
            key, subkey = jr.split(key)
            self.cond_linear.append(
                eqx.nn.Linear(cond_dim, first_layer_out_dim*2, use_bias=True, key=subkey)
            )
        else:
            self.cond_linear = None

        self.depth = depth
        self.layers = layers_and_log_jac_fns
        self.block_dim = block_dim
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        self.activation = activation
        self.activation_tanh=Tanh()
        self.beta=1e-15
        self.L_x=jax.numpy.tril(jnp.ones((dim,dim)),k=-1)*self.beta+jnp.eye(dim)
        self.U_x=jax.numpy.triu(jnp.ones((dim,dim)),k=1)*self.beta+jnp.eye(dim)
        self.L_c=jax.numpy.tril(jnp.ones((cond_dim,cond_dim)),k=-1)*self.beta+jnp.eye(cond_dim)
        self.U_c=jax.numpy.triu(jnp.ones((cond_dim,cond_dim)),k=1)*self.beta+jnp.eye(cond_dim)


    def transform_and_log_det(self, x, condition=None):
        log_dets_3ds = []

        #x=self.L_x@(self.U_x@x)
        #condition=self.L_c@(self.U_c@condition)

        for i, (linear, log_jacobian_fn) in enumerate(self.layers[:-1]):
            x = linear(x)
            
            # Only apply conditioning on the first layer to prevent over-expressiveness
            if i == 0:
                out_condition=self.cond_linear[i](condition)
                assert out_condition.shape[-1]%2 == 0
                
                # Constrained conditioning: prevent delta function collapse
                half_dim = out_condition.shape[-1] // 2
                
                # Constrain scale to reasonable range [0.1, 3.0] to prevent collapse/explosion
                raw_scale = out_condition[..., :half_dim]
                scale_term = 0.1 + 2.9 * jax.nn.sigmoid(raw_scale)  # Maps to [0.1, 3.0]
                shift_term = out_condition[..., half_dim:]
                
                # Apply constrained conditioning
                x = x * scale_term + shift_term
            
            if(i==0):
                activation=self.activation
            else:
                activation=self.activation_tanh

            log_dets_3ds.append(log_jacobian_fn(linear))
            x, log_det_3d = self._activation_and_log_jacobian_3d(activation,x)
            log_dets_3ds.append(log_det_3d)

        linear, log_jacobian_fn = self.layers[-1]
        x = linear(x)
        log_dets_3ds.append(log_jacobian_fn(linear))

        log_det = log_dets_3ds[-1]
        for log_jacobian in reversed(log_dets_3ds[:-1]):
            log_det = logmatmulexp(log_det, log_jacobian)
        return x, log_det.sum()

    def inverse_and_log_det(self, y, condition=None):
        raise NotImplementedError(
            "Inverse methods not implemented for block autoregressive network. "
            "Consider providing a numerical method with ``NumericalInverse``. "
            "See ``flowjax.root_finding`` for numerical methods."
        )

    def _activation_and_log_jacobian_3d(self,activation, x):
        """Compute activation and the log determinant (blocks, block_dim, block_dim)."""
        x, log_abs_grads = eqx.filter_vmap(activation.transform_and_log_det)(x)
        log_det_3d = jnp.full((self.shape[0], self.block_dim, self.block_dim), -jnp.inf)
        diag_idxs = jnp.arange(self.block_dim)
        log_det_3d = log_det_3d.at[:, diag_idxs, diag_idxs].set(
            log_abs_grads.reshape(self.shape[0], self.block_dim)
        )
        return x, log_det_3d


def block_autoregressive_linear(
    key: PRNGKeyArray,
    *,
    n_blocks: int,
    block_shape: tuple,
) -> tuple[eqx.nn.Linear, Callable]:
    """Block autoregressive linear layer (https://arxiv.org/abs/1904.04676).

    Returns:
        Tuple containing 1) an equinox.nn.Linear layer, with the weights wrapped
        in order to provide a block lower triangular mask and weight normalisation.
        2) callable taking the linear layer, returning the log block diagonal weights.

    Args:
        key: Random key.
        n_blocks: Number of diagonal blocks (dimension of original input).
        block_shape: The shape of the blocks.
    """
    out_features, in_features = (b * n_blocks for b in block_shape)
    linear = eqx.nn.Linear(in_features, out_features, key=key)
    block_diag_mask = masks.block_diag_mask(block_shape, n_blocks)
    block_tril_mask = masks.block_tril_mask(block_shape, n_blocks)

    def apply_mask(weight):
        weight = jnp.where(block_tril_mask, weight, 0)
        return jnp.where(block_diag_mask, softplus(weight), weight)

    weight = WeightNormalization(Parameterize(apply_mask, linear.weight))
    linear = eqx.tree_at(lambda linear: linear.weight, linear, replace=weight)

    def linear_to_log_block_diagonal(linear: eqx.nn.Linear):
        idxs = jnp.where(block_diag_mask, size=prod(block_shape) * n_blocks)
        jac_3d = linear.weight[idxs].reshape(n_blocks, *block_shape)
        return jnp.log(jac_3d)

    return linear, linear_to_log_block_diagonal


def logmatmulexp(x, y):
    """Numerically stable version of ``(x.log() @ y.log()).exp()``.

    From numpyro https://github.com/pyro-ppl/numpyro/blob/
    f2ff89a3a7147617e185eb51148eb15d56d44661/numpyro/distributions/util.py#L387.
    """
    x_shift = jax.lax.stop_gradient(jnp.amax(x, -1, keepdims=True))
    y_shift = jax.lax.stop_gradient(jnp.amax(y, -2, keepdims=True))
    xy = jnp.log(jnp.matmul(jnp.exp(x - x_shift), jnp.exp(y - y_shift)))
    return xy + x_shift + y_shift