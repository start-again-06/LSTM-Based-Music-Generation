import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
with open("Dataset/Music_file.txt", "r", encoding="utf-8") as f:
    songs = f.read().split("\n\n")

print(f"Loaded {len(songs)} songs")

example_song = songs[0]
print("\nExample song:\n")
print(example_song)
songs_joined = "\n\n".join(songs)
vocab = sorted(set(songs_joined))
print(f"\nVocabulary size: {len(vocab)}")
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

print("\nFirst 20 character mappings:")
for char in list(char2idx.keys())[:20]:
    print(f"{repr(char)} → {char2idx[char]}")
    vocab = sorted(set(songs))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def vectorize_string(string):
    return np.array([char2idx[c] for c in string])

vectorized_songs = vectorize_string(songs)
def get_batch(vectorized_data, seq_length, batch_size):
    n = len(vectorized_data) - 1
    idx = np.random.choice(n - seq_length, batch_size)

    input_batch = [vectorized_data[i:i+seq_length] for i in idx]
    target_batch = [vectorized_data[i+1:i+seq_length+1] for i in idx]

    x = torch.tensor(input_batch, dtype=torch.long)
    y = torch.tensor(target_batch, dtype=torch.long)

    return x, y
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.hidden_size).to(device),
            torch.zeros(1, batch_size, self.hidden_size).to(device)
        )

    def forward(self, x, state=None):
        x = self.embedding(x)

        if state is None:
            state = self.init_hidden(x.size(0), x.device)

        out, state = self.lstm(x, state)
        out = self.fc(out)

        return out, state
params = {
    "embedding_dim": 256,
    "hidden_size": 512,
    "batch_size": 32,
    "seq_length": 100,
}

model = LSTMModel(
    len(vocab),
    params["embedding_dim"],
    params["hidden_size"]
).to(device)

print(model)
x, y = get_batch(vectorized_songs, params["seq_length"], params["batch_size"])
x = x.to(device)
y = y.to(device)

pred, _ = model(x)

print("Input shape:", x.shape)
print("Prediction shape:", pred.shape)
probs = torch.softmax(pred[0], dim=-1)
sampled_indices = torch.multinomial(probs, num_samples=1)
sampled_indices = sampled_indices.squeeze(-1).cpu().numpy()
input_text = "".join(idx2char[x[0].cpu()])
predicted_text = "".join(idx2char[sampled_indices])

print("Input:\n", repr(input_text))
print("\nNext Char Predictions:\n", repr(predicted_text))
print("Input: \n", repr("".join(idx2char[x[0].cpu()])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
loss_fn = nn.CrossEntropyLoss()

def compute_loss(labels, logits):
    labels: (batch_size, sequence_length)
    logits: (batch_size, sequence_length, vocab_size)

    Returns:
        Scalar cross-entropy loss over the batch
    labels = labels.view(-1)
    logits = logits.view(-1, logits.size(-1))
    return loss_fn(logits, labels)
    example_batch_loss = compute_loss(y, pred)

print(f"Prediction shape: {pred.shape}")
print(f"Initial loss (untrained model): {example_batch_loss.item():.4f}")
params = {
    "num_training_iterations": 2000,
    "batch_size": 32,
    "seq_length": 100,
    "learning_rate": 3e-4,
    "embedding_dim": 256,
    "hidden_size": 512,
}

vocab_size = len(vocab)

model = LSTMModel(
    vocab_size,
    params["embedding_dim"],
    params["hidden_size"]
).to(device)

optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

loss_history = []
model = LSTMModel(
    vocab_size,
    params["embedding_dim"],
    params["hidden_size"]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params["learning_rate"]
)

loss_history = []
for iteration in tqdm(range(params["num_training_iterations"])):
    x_batch, y_batch = get_batch(
        vectorized_songs,
        params["seq_length"],
        params["batch_size"]
    )
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    model.train()
    optimizer.zero_grad()
    logits, _ = model(x_batch)
    loss = compute_loss(y_batch, logits)
    loss.backward()
     optimizer.step()

    loss_history.append(loss.item())

    if iteration % 100 == 0:
        print(f"Iteration {iteration} | Loss: {loss.item():.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
def generate_text(model, start_string, generation_length=500):
    model.eval()

    # Convert start string to indices
    input_indices = torch.tensor(
        [char2idx[c] for c in start_string],
        dtype=torch.long
    ).unsqueeze(0).to(device)

    state = None
    generated = start_string

    for _ in range(generation_length):
        logits, state = model(input_indices, state)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        next_char = idx2char[next_idx.item()]
        generated += next_char
        input_indices = next_idx.unsqueeze(0)

    return generated
generated_text = generate_text(
    model,
    start_string="X:1\n",
    generation_length=800
)

print("\nGenerated Text:\n")
print(generated_text)








