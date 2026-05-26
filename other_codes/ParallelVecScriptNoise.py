import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Dataset

torch.random.manual_seed(42)
unlabeled = 20
num_classes = 2
label_budget = 8

data = torch.empty(unlabeled, num_classes).uniform_(-1, 1)
# Noise margin
margin = 1.0
distance = (data[:, 1] - data[:, 0]) / (2 ** 0.5)  # distance from the decision boundary
labels = torch.where(
    distance.abs() < margin,
    torch.bernoulli(torch.full((unlabeled,), 0.5)),  # in margin
    (distance >= 0).float()                          # outside of margin
).unsqueeze(1)

all_subsets = list(combinations(range(unlabeled), label_budget))
x_all = torch.stack([data[list(subset)] for subset in all_subsets])
y_all = torch.stack([labels[list(subset)] for subset in all_subsets])

# x @ W.T + b
N = len(all_subsets)
D_in = num_classes
H = 0 # hidden layer size - not used
D_out = 1
W = torch.randn(N, D_out, D_in, requires_grad=True)
b = torch.randn(N, D_out, requires_grad=True)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD([W, b], lr=0.1)

# Training loop
num_epochs = 100
loss_history = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    logits = x_all @ W.transpose(-1, -2) + b.unsqueeze(1)
    loss = criterion(logits, y_all)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
print(f'Final Loss: {loss_history[-1]:.4f}')

# Test Dataset
test_data = torch.cartesian_prod(torch.linspace(-1, 1, 20), torch.linspace(-1, 1, 20))
test_labels = (test_data[:, 1] >= test_data[:, 0]).float().unsqueeze(1)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=400, shuffle=False)

# Evaluation
with torch.no_grad():
    for test_x, test_y in test_loader:
        test_logits = test_x.unsqueeze(0) @ W.transpose(-1, -2) + b.unsqueeze(1)
        test_preds = (test_logits >= 0).float()
        accuracies =(test_preds == test_y.unsqueeze(0)).float().mean(dim=[1, 2])
        
print(f'Test Accuracy: {accuracies.mean().item():.4f}')

# Visualization of NN accuracy distribution across all subsets
plt.figure(figsize=(18, 12))

n, bins, patches = plt.hist(accuracies.numpy(), bins=50, range=(0, 1), edgecolor='black', color='steelblue', alpha=0.5)

for i in range(len(patches)):
    count = n[i]
    if count >0:
        x = patches[i].get_x() + patches[i].get_width() / 2
        y = patches[i].get_height()
        plt.text(x, y + (max(n)*0.01), str(int(count)),
                 ha='center', va='bottom', fontsize=12, rotation=60)
        
plt.axvline(np.mean(accuracies.numpy()), color='blue', linestyle='--', label=f'Mean: {np.mean(accuracies.numpy()):.4f}')
plt.xlabel('Accuracy')
plt.ylabel('Number of Subsets')
plt.title(f'Distribution of NN Accuracies Across All {len(all_subsets)} (n={unlabeled}, k={label_budget}) Subsets w/ Noise={margin}, Loss={loss_history[-1]:.4f}')
plt.legend(fontsize=24)
plt.tight_layout()
plt.savefig(f'Distribution {len(all_subsets)}, n={unlabeled}, k={label_budget} with Counts, Noise={margin}.png')

# RF w/ noise
accuracies_rf = []
count = 0
print(f"--- Random Forest Training and Evaluating for All {len(all_subsets)} Subsets ---")
for idx in range(len(all_subsets)):
    subset_idx = all_subsets[idx]
    x_train_rf = data[list(subset_idx)].cpu().numpy()
    y_train_rf = labels[list(subset_idx)].flatten().cpu().numpy()
    
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    rf_model.fit(x_train_rf, y_train_rf)
    
    x_test_np = test_data.cpu().numpy()
    y_test_np = test_labels.flatten().cpu().numpy()

    rf_preds = rf_model.predict(x_test_np)
    rf_acc = accuracy_score(y_test_np, rf_preds)
    accuracies_rf.append(rf_acc)
    count += 1
    if count % 1000 == 0:
        print(f" {count} / {len(all_subsets)} completed")


# visualize RF accuracy distribution across all subsets w/ noise
plt.figure(figsize=(18, 12))
n, bins, patches = plt.hist(accuracies_rf, bins=50, range=(0, 1), edgecolor='black', color='sandybrown', alpha=0.7)

# get the count for each bar and show it above the bar
for i in range(len(patches)):
    count = n[i]
    if count > 0:
        x_pos = patches[i].get_x() + patches[i].get_width() / 2
        y_pos = patches[i].get_height()
        plt.text(x_pos, y_pos + (max(n) * 0.01), str(int(count)), 
                 ha='center', va='bottom', fontsize=12, rotation=60)

plt.axvline(np.mean(accuracies_rf), color='blue', linestyle='--', label=f'Mean: {np.mean(accuracies_rf):.4f}')
plt.xlabel('Test Accuracy')
plt.ylabel('Number of Subsets')
plt.title(f'Random Forest Performance Distribution {len(all_subsets)} Subsets (n={unlabeled}, k={label_budget}), Noise={margin}')
plt.legend(fontsize=24)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()

plt.savefig(f'RF_Accuracy_Distribution_n{unlabeled}_k{label_budget}_noise{margin}.png')
plt.show()

# visualization of NN vs RF accuracy distribution w/ noise
fig, ax = plt.subplots(figsize=(10, 5))
common_range = (0, 1)
common_bins = 50
ax.hist(accuracies.numpy(), bins=common_bins, range=common_range, alpha=0.5, edgecolor='black', label='Neural Network')
ax.hist(accuracies_rf,      bins=common_bins, range=common_range, alpha=0.5, edgecolor='black', label='Random Forest')
ax.set_xlabel('Accuracy')
ax.set_ylabel('Number of Subsets')
ax.set_title(f'NN vs Random Forest: Distribution of Test Accuracy {len(all_subsets)} (n={unlabeled}, k={label_budget}), noise={margin}')
ax.legend()

table_data = [
    [f'{accuracies.max():.4f}',  f'{np.max(accuracies_rf):.4f}'],
    [f'{accuracies.mean():.4f}', f'{np.mean(accuracies_rf):.4f}'],
    [f'{accuracies.min():.4f}',  f'{np.min(accuracies_rf):.4f}'],
]

table = ax.table(
    cellText=table_data,
    rowLabels=['Max', 'Mean', 'Min'],
    colLabels=['NN', 'RF'],
    loc='top left',        
    cellLoc='center',        
    bbox=[0.06, 0.78, 0.2, 0.2]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(0.3, 0.6) 

plt.tight_layout()
plt.savefig(f'NN_vs_RF_distribution(n={unlabeled}, k={label_budget}, noise={margin}, {len(all_subsets)}).png')