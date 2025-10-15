import jax
import os
import numpy as np
#import sbibm
import pickle
import pandas as pd
import jax.numpy as jnp
from sklearn.datasets import make_moons
from jax import jit, vmap, lax
from jax import random
from utils import CS,SIR
from typing import Tuple, Optional


def get_dataset(config, evaluation=False,load_data=True,return_summary=False):
    """Load a dataset based on config settings."""
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch size ({batch_size}) must be divisible by the number of devices ({jax.device_count()})')
    
    per_device_batch_size = batch_size 
    num_epochs = None if not evaluation else 1
    batch_dims = [ per_device_batch_size] 
    dataset_name = config.data.dataset
    num_simulations=config.data.num_simulations
    if(return_summary):
        train_ds, eval_ds,data_mean,data_std,train_stats_mean,train_stats_std = load_custom_dataset(dataset_name, config,num_simulations,load_data,return_summary)
        key, subkey = random.split(random.PRNGKey(0))
        train_ds = preprocess_and_batch(train_ds, batch_dims, key, evaluation)
        key, subkey = random.split(key)
        eval_ds = preprocess_and_batch(eval_ds, batch_dims, key, evaluation)
        return train_ds, eval_ds,data_mean,data_std,train_stats_mean,train_stats_std
    else:
        key, subkey = random.split(random.PRNGKey(0))
        train_ds, eval_ds = load_custom_dataset(dataset_name, config,num_simulations,load_data)
        train_ds = preprocess_and_batch(train_ds, batch_dims, key, evaluation)
        key, subkey = random.split(key)
        eval_ds = preprocess_and_batch(eval_ds, batch_dims, key, evaluation)
        return train_ds, eval_ds


def get_inference_dataset(config, evaluation=False):
    """Load a dataset based on config settings."""
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    num_simulations=config.data.inference_simulations
    if batch_size % jax.device_count() != 0:
        raise ValueError(f'Batch size ({batch_size}) must be divisible by the number of devices ({jax.device_count()})')
    
    per_device_batch_size = batch_size 
    num_epochs = None if not evaluation else 1
    batch_dims = [ per_device_batch_size] 
    inference_dataset_name = config.data.inference_dataset


    train_ds, eval_ds = load_custom_dataset(inference_dataset_name, config,num_simulations,load_data=True,return_summary=False,inference=True) #return summary always false for this, load data alwyas true
    key, subkey = random.split(random.PRNGKey(0))
    train_ds = preprocess_and_batch(train_ds, batch_dims, key, evaluation)
    key, subkey = random.split(key)
    eval_ds = preprocess_and_batch(eval_ds, batch_dims, key, evaluation)
    return train_ds, eval_ds
    
    

def load_custom_dataset(dataset_name, config,num_simulations,load_data=True,return_summary=False,inference=False):
    """Load custom dataset from external files or generate it if needed.
    
    This function loads a dataset from external files if they exist. If the files do not exist,
    it generates the dataset using the specified configuration. The function also computes the 
    mean and standard deviation of the dataset for normalization purposes. The standardized data 
    is only saved if it's training data (inference=False); otherwise, the regular, unscaled data 
    is saved. The function returns the training and evaluation datasets, and optionally the mean 
    and standard deviation if return_summary is True.
    """
    data_path = os.path.join(config.data.data_path, f"{dataset_name}.npy")
    stats_path_pkl = os.path.join(config.data.data_path, f"{dataset_name}_stats.npy")
    stats_path_npz = os.path.join(config.data.data_path, f"{dataset_name}_stats.npz")

    if os.path.exists(data_path):
        # Try loading stats from .npz first, then fall back to .npy
        if os.path.exists(stats_path_npz):
            stats = np.load(stats_path_npz)
            mean, std = stats['mean'], stats['std']
            # Load train_stats for flow normalization if available
            train_stats_mean = stats.get('train_stats_mean', None)
            train_stats_std = stats.get('train_stats_std', None)
        elif os.path.exists(stats_path_pkl):
            with open(stats_path_pkl, 'rb') as f:
                stats = pickle.load(f)  # Load saved stats if they exist
            mean, std = stats['mean'], stats['std']
            # Load train_stats for flow normalization if available
            train_stats_mean = stats.get('train_stats_mean', None)
            train_stats_std = stats.get('train_stats_std', None)
        else:
            raise FileNotFoundError(f"Neither {stats_path_npz} nor {stats_path_pkl} found")
        if(load_data==False):
            if(return_summary):
                data=jnp.zeros((1,config.data.summary_dim+config.model.cond_dim+config.data.vector_dim))
            else:
                raise ValueError("Data not loaded, must have return_summary= False then load_data=True")
        else:
            data = np.load(data_path, allow_pickle=True)
    else:
        if(config.data.benchmark==True): #legacy sbibm loading
            data = None           
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
            # No train_stats available for generated data
            train_stats_mean = None
            train_stats_std = None
        else:
            data = generate_other_data(config,dataset_name,num_simulations)
            # Compute mean and std for training data (joint_data)
            mean = np.mean(data[:, :], axis=0)
            std = np.std(data[:, :], axis=0)
            # No train_stats available for generated data
            train_stats_mean = None
            train_stats_std = None
        if(inference==False):
            # we only scale data for the training, not for the inference!
            np.save(data_path, (data-mean)/std) 
        else:
            assert inference==True #"Inference must be true"
            np.save(data_path, data)
        
        # Save mean and std to a file
        #with open(stats_path, 'wb') as f:
        #    pickle.dump({'mean': mean, 'std': std}, f)


    dataset=data
    n = len(dataset)

    # Split dataset into training and evaluation sets
    if not inference:
        p = config.training.validation_split
    else:
        p = 1  # Use entire dataset as eval

    split_idx = int((1 - p) * n)

    train_ds = dataset[:split_idx]
    eval_ds = dataset[split_idx:]

    if return_summary:
        return train_ds, eval_ds, mean, std, train_stats_mean, train_stats_std
    else: 
        return train_ds, eval_ds



def generate_other_data(config,dataset_name,num_simulations):
    """Generate data using other task and simulator, with conditional prior sampling."""
   
    return None





def preprocess_and_batch(
    dataset: jax.Array,      # Single JAX array (e.g., shape [num_samples, ...])
    batch_dims: Tuple[int, ...],  # E.g., (32, 256) for 2D batching
    key: Optional[jax.random.PRNGKey] = None,  # Optional for shuffling
    inference: bool = False,
) -> jax.Array:
    """
    JAX-compatible batching for a single array dataset.
    
    Args:
        dataset: Input JAX array of shape (num_samples, ...)
        batch_dims: Tuple of batch sizes (outermost first)
        key: PRNG key for shuffling (ignored if inference=True)
        inference: If True, skip shuffling
        
    Returns:
        Batched array with shape (batch_dim1, batch_dim2, ...)
    """
    # --- Step 1: Shuffle (if not inference) ---
    if inference:
      return dataset
    
    dataset = jax.random.permutation(key, dataset, axis=0)

    # --- Step 2: Batch along each dimension ---
    for batch_size in reversed(batch_dims):
        num_samples = dataset.shape[0]
        num_batches = num_samples // batch_size
        dataset = dataset[:num_batches * batch_size]  # Trim if needed
        dataset = dataset.reshape(num_batches, batch_size, *dataset.shape[1:])

    return dataset