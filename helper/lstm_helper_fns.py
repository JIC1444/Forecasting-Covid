# General.
import os
import glob
import datetime
import math
import random
import re

# Data manipulation.
import pandas as pd
import numpy as np
import geopandas as gpd
from libpysal import weights
import networkx as nx
import matplotlib.pyplot as plt

# Torch.
import torch
import torch.nn as nn
import torch.nn.functional as F


def display_preds(label, output, loss):
    """
    A function that displays multiple sample predictions in subplots.

    Args:
    - label: np.array, shape (N, 7) or (7,) if single sample.
    - output: np.array, shape (N, 7) or (7,) if single sample.
    - loss: float, loss value to display in the title.
    
    Displays a plot (or a series of subplots) with Actual and Predicted Cases.
    """
    x = np.arange(1, 8, 1)
    N = output.shape[0]
    fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(8, 3 * N), sharex=True, sharey=True)
        
    if N == 1:
        axes = [axes]
        
    for i in range(N):
        axes[i].plot(x, label[i], 'b', label='Actual Cases')
        axes[i].plot(x, output[i], 'r', label='Predicted Cases')
        axes[i].axis([1, 7, 0, 1])
        axes[i].set_ylabel('Normalized Cases')
        axes[i].set_title(f'Sample {i+1}; Loss: {loss}')
        axes[i].legend()
        
    axes[-1].set_xlabel('Day 21-28')
    plt.tight_layout()
    plt.show()

def isnans(M):
    # Returns bool if an nd tensor contains a nan value.
    return True if any(torch.isnan(M.flatten())) else False

def detach_np(M):
    return M.detach().cpu().numpy()

def create_weights_folder(f):
    """
    Creates a folder with the next sequential name, e.g if LSTM01 exists, LSTM02 folder 
    will be created.

    Args:
    - f: str, the filepath to the weights folder.

    Returns the current folder that the weights will be saved in.
    """
    existing = os.listdir(f)
    if len(existing) != 0:
        fs = sorted(existing)
        lastnum = int(fs[-1][-2:]) + 1
        new_folder_name = f"LSTM{lastnum:02d}"

    else: # If no folders currently exist in the weights path.
        new_folder_name = "LSTM00"
    fpath = os.path.join(f, new_folder_name)
    os.makedirs(fpath, exist_ok=False) # Do NOT want to replace results.
    return fpath

def plot_accs(loss1, loss2=None, loss3=None):
    """
    Plots the model progress over the epochs.

    Args:
    - loss1: np.array of the losses from each epoch.
    - loss2: (optional) np.array of the losses from each epoch.

    Returns a plot of the losses over all the epochs.
    """
    plt.plot(np.arange(1, loss1.shape[0]+1, 1), loss1, label='Training')
    if loss2 is not None:
        plt.plot(np.arange(1, loss2.shape[0]+1, 1), loss2, label='Validation')
    if loss3 is not None:
        plt.plot(np.arange(1, loss3.shape[0]+1, 1), loss3, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('MSELoss')
    plt.title('MSE loss over epochs')
    plt.legend()
    plt.show()


def test_entire_set(model, df, window_size, batch_size):
    """
    Plots the model's performance over the entire dataset.

    Args:
    - model: torch.nn.Module.
    - df: pd.DataFrame, the entire dataset (can be training, validation and testing
      or just validation and testing)
    - window_size: int, the input size of the model.
    - batch_size: int.
    """
    model.eval()
    total_loss = 0.0

    # Turn a numerical dataframe into a tensor to loop over.
    data = torch.tensor(df.values, dtype=torch.float32)
    num_rows = data.shape[0]
    zeros = torch.zeros((batch_size-num_rows, data.shape[1]))
    data = torch.concat([data, zeros], dim=0) # Pad the tensor to fit batch size.
    
    data_shape = data.shape
    actual_data = data[:, window_size:].numpy()
    results = np.zeros_like(actual_data) # The first window_size days cannot be tested.
    times_covered = np.zeros_like(results)

    for col in range(data_shape[1]-window_size-1):
        # Indicies.
        in_st, in_end = col, col+window_size-7
        lab_st, lab_end = in_end+1, col+window_size+1

        # Take the input window-sized day sample.
        input = data[:, in_st:in_end]
        label = data[:, lab_st:lab_end]

        with torch.no_grad():
            output = model(input)
            # Store output in the rows for each county.
            results[:, lab_st:lab_end] += output.detach().cpu().numpy()
            times_covered[:, lab_st:lab_end] += 1
        
    # Average by how many times the array was covered.
    times_covered[times_covered == 0] = 1
    results[results == 0] = 1
    results = np.divide(results, times_covered)

    x = np.arange(0, results.shape[1], 1)
    plt.plot(x, np.mean(results[:num_rows, :], axis=0), label='Predicted')
    plt.plot(x, np.mean(actual_data[:num_rows, :], axis=0), label='Actual')
    plt.axis([0, x[-1], 0, 1])
    plt.legend()
    plt.show()
    return results


def load_model_list(paths, device):
    model_list = []
    for p in paths:
        model = torch.load(p, map_location=device, weights_only=False)
        model_list.append(model)
    return model_list
