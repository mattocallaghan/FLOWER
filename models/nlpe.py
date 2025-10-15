"""Premade versions of common flow architetctures from ``flowjax.flows``.

All these functions return a :class:`~flowjax.distributions.Transformed` distribution.

I have adapted the flows very slightly here
"""
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import Linear
from jax.nn import softplus
from jax.nn.initializers import glorot_uniform
from jaxtyping import Array, PRNGKeyArray
from paramax import Parameterize, WeightNormalization
from paramax.utils import inv_softplus

from flowjax.bijections import (
    AbstractBijection,
    AdditiveCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    Coupling,
    Flip,
    Invert,
    LeakyTanh,
    MaskedAutoregressive,
    NumericalInverse,
    Permute,
    Planar,
    RationalQuadraticSpline,
    Sandwich,
    Scan,
    TriangularAffine,
    Vmap,
)
from flowjax.distributions import AbstractDistribution, Transformed
from flowjax.root_finding import (
    bisect_check_expand_search,
    root_finder_to_inverter,
)
from models.bnaf_elu import BNAF_ELU
from flowjax.distributions import AbstractDistribution, Transformed
#from flowjax.utils import inv_softplus
#from flowjax.wrappers import Parameterize, WeightNormalization




def nlpe_bnaf_elu(
    key: PRNGKeyArray,
    *,
    base_dist: AbstractDistribution,
    cond_dim: int | None = None,
    nn_depth: int = 1,
    nn_block_dim: int = 8,
    flow_layers: int = 1,
    invert: bool = True,
    activation: AbstractBijection | Callable | None = None,
    inverter: Callable[[AbstractBijection, Array, Array | None], Array] | None = None,
) -> Transformed:
    """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).

    This is identical to the flowjax formulation, but I call the different bijection from block_bijection.py

    Each flow layer contains a
    :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`
    bijection. The bijection does not have an analytic inverse, so must be inverted
    using numerical methods (by default a bisection search). Note that this means
    that only one of ``log_prob`` or ``sample{_and_log_prob}`` can be efficient,
    controlled by the ``invert`` argument. Note, ensuring reasonably scaled base and
    target distributions will be beneficial for the efficiency of the numerical inverse.

    Args:
        key: Jax key.
        base_dist: Base distribution, with ``base_dist.ndim==1``.
        cond_dim: Dimension of conditional variables. Defaults to None.
        nn_depth: Number of hidden layers within the networks. Defaults to 1.
        nn_block_dim: Block size. Hidden layer width is dim*nn_block_dim. Defaults to 8.
        flow_layers: Number of BNAF layers. Defaults to 1.
        invert: Use `True` for efficient ``log_prob`` (e.g. when fitting by maximum
            likelihood), and `False` for efficient ``sample`` and
            ``sample_and_log_prob`` methods (e.g. for fitting variationally).
        activation: Activation function used within block neural autoregressive
            networks. Note this should be bijective and in some use cases should map
            real -> real. For more information, see
            :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`.
            Defaults to :class:`~flowjax.bijections.tanh.LeakyTanh`.
        inverter: Callable that implements the required numerical method to
            invert the ``BlockAutoregressiveNetwork`` bijection. Passed to
            :py:class:`~flowjax.bijections.NumericalInverse`. Defaults to
            using ``elementwise_autoregressive_bisection``.
    """
    dim = base_dist.shape[-1]

    if inverter is None:
        inverter = root_finder_to_inverter(
            partial(
                bisect_check_expand_search,
                midpoint=jnp.zeros(dim),
                width=5,
            )
        )

    def make_layer(key):  # bnaf layer + permutation
        bij_key, perm_key = jr.split(key)
        bijection = NumericalInverse(
            BNAF_ELU(
                bij_key,
                dim=base_dist.shape[-1],
                cond_dim=cond_dim,
                depth=nn_depth,
                block_dim=nn_block_dim,
                activation=activation,
            ),
            inverter=inverter,
        )
        return _add_default_permute(bijection, base_dist.shape[-1], perm_key)

    keys = jr.split(key, flow_layers)
    layers = eqx.filter_vmap(make_layer)(keys)
    bijection = Invert(Scan(layers)) if invert else Scan(layers)
    return Transformed(base_dist, bijection)



def _add_default_permute(bijection: AbstractBijection, dim: int, key: PRNGKeyArray):
    if dim == 1:
        return bijection
    if dim == 2:
        return Chain([bijection, Flip((dim,))]).merge_chains()

    perm = Permute(jr.permutation(key, jnp.arange(dim)))
    return Chain([bijection, perm]).merge_chains()