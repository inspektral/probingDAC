import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
import soundfile as sf
import dac
from audiotools import AudioSignal

def audio_file_to_tensor(file_path, to_mono=True):
    audio, sr = sf.read(file_path)
    if to_mono and len(audio.shape) > 1:
        audio = audio.sum(axis=1) / 2
    audio = torch.tensor(audio, dtype=torch.float32)
    return audio, sr

def encode_and_decode(model, audio, sr, device):
    snd = AudioSignal(audio, sr)

    snd.to(model.device)
    snd_x = model.preprocess(snd.audio_data, snd.sample_rate)

    with torch.no_grad():
        snd_z, snd_codes, snd_latents, _, _ = model.encode(snd_x)
        snd_decoded = model.decode(snd_z)

    audio = snd_decoded[0,0,:].cpu().detach().numpy()
    return audio

def sine_sweep(f0, f1, duration, sr):
    t = np.linspace(0, duration, int(duration*sr), endpoint=False)
    freqs = np.logspace(np.log10(f0), np.log10(f1), num=len(t), endpoint=True)
    signal = np.sin(np.pi * freqs * t)
    return signal

def white_noise(duration, sr):
    return np.random.randn(int(duration*sr))

def plot_waveform(audio, sr):
    plt.style.use('dark_background')
    plt.figure(figsize=(15, 6))
    time = np.arange(len(audio)) / sr
    plt.plot(time, audio, linewidth=1, color='#00ff00')
    plt.xlabel('Time (seconds)', color='white')
    plt.ylabel('Amplitude', color='white')
    plt.title('Audio Waveform', color='white')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_heatmap(snd):
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    sns.heatmap(snd.cpu().numpy()[0,:,:], cmap='viridis')
    plt.xlabel('Time Frame', color='white')
    plt.ylabel('Dimension', color='white')
    plt.title('Latent space Heatmap', color='white')
    plt.tight_layout()
    plt.show()

def plot_lines(snd):
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    
    # Get data and transpose to plot each dimension
    data = snd.cpu().numpy()[0,:,:]
    time_steps = np.arange(data.shape[1])
    
    # Plot each dimension
    for i in range(data.shape[0]):
        plt.plot(time_steps, data[i,:], alpha=0.5, linewidth=0.5)
    
    plt.grid(True, alpha=0.2)
    plt.xlabel('Time Frame', color='white')
    plt.ylabel('Amplitude', color='white')
    plt.title('Latent Space Time Series', color='white')
    plt.tight_layout()
    plt.show()

lol  = 0

def plot_lines_separated(snd):
    data = snd.cpu().numpy()[0,:,:]
    n_dims = data.shape[0]
    
    # Calculate grid dimensions
    n_cols = 1  # Fixed number of columns
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    plt.style.use('dark_background')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Find global min and max for consistent y-axis scaling
    y_min = data.min()
    y_max = data.max()
    
    # Plot each dimension
    for i in range(n_dims):
        axes[i].plot(data[i,:], linewidth=0.8, color='cyan')
        axes[i].grid(True, alpha=0.2)
        axes[i].set_title(f'Dimension {i}', color='white', fontsize=8)
        axes[i].set_ylim(y_min, y_max)  # Set consistent y-axis limits
        
    
    plt.title('Latent Space Dimensions', color='white', fontsize=16)
    plt.tight_layout()
    plt.show()


