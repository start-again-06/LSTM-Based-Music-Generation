import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
with open("", "r", encoding="utf-8") as f:
    songs = f.read().split("\n\n")

print(f"Loaded {len(songs)} songs")

# Inspect one example
example_song = songs[0]
print("\nExample song:\n")
print(example_song)
     
