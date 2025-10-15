import jax.numpy as jnp
from jax import random
import jax
from tensorflow_probability.substrates import jax as tfp
import functools
tfd = tfp.distributions
tfb=tfp.bijectors
tfpk = tfp.math.psd_kernels
tfpe=tfp.experimental
import flowjax
import abc

import inference_models.inference_flower as inference_flower_task
import jax
import time
from datetime import datetime


Root = tfd.JointDistributionCoroutine.Root
#import sbibm
import numpy as np
#potentially do the flowjax way
_INFERENCES = {}
mcmc = tfp.mcmc

def register_inference(cls=None, *, name=None):
    """A decorator for registering inference problem classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _INFERENCES:
            raise ValueError(f'Already registered inference model with name: {local_name}')
        _INFERENCES[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register

def get_inference(name):
    """Retrieve a registered inference class by name."""
    return _INFERENCES[name]









########################################################################################
########################################################################################
# Flower
########################################################################################
########################################################################################

register_inference(inference_flower_task.FLower_Embedding_NLPE,name='flower')