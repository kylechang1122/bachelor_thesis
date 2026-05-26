import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.random.manual_seed(42)
data = torch.empty(20, 2).uniform_(-1, 1)
labels = (data[:, 1] >= data[:, 0]).float().unsqueeze(1)
dataset = TensorDataset(data, labels)

# Fixed subset of 4 samples for training
indices = [9, 12, 13, 17]
subset = torch.utils.data.Subset(dataset, indices=indices)
loader = DataLoader(subset, batch_size=4)

model = nn.Linear(2, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

losses = []
for epoch in range(100):
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())

print(f"Final loss: {loss.item():.4f}")

# Test dataset: 40 points evenly spread over the [-1, 1]^2 plane (8x5 grid)
test_data = torch.cartesian_prod(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 20))
test_labels = (test_data[:, 1] >= test_data[:, 0]).float().unsqueeze(1)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=400)

model.eval()
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x)
        preds = (logits >= 0).float()
        accuracy = (preds == y).float().mean()

print(f"Test accuracy: {accuracy.item():.4f}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

correct = (preds == y).squeeze().numpy()
x_np = x.numpy()

fig, (ax_loss, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))

ax_loss.plot(losses)
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Training loss")

ax_scatter.scatter(x_np[correct, 0],  x_np[correct, 1],  c="green", marker="o", label="Correct")
ax_scatter.scatter(x_np[~correct, 0], x_np[~correct, 1], c="red",   marker="x", label="Incorrect")
ax_scatter.plot([-1, 1], [-1, 1], color="gray", linestyle="--", linewidth=0.8, label="Decision boundary (x₁=x₀)")
ax_scatter.set_xlabel("x₀")
ax_scatter.set_ylabel("x₁")
ax_scatter.set_title(f"Test set classification results ({','.join(map(str, indices))})")
ax_scatter.legend()

plt.tight_layout()
#plt.show()
plt.savefig(f"result ({','.join(map(str, indices))}).png")
print(f"saved as result ({','.join(map(str, indices))}).png")