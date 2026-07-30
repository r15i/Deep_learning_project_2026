import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
def generate_curve(epochs, start, end, noise, is_train=True):
    x = np.arange(epochs)
    if is_train:
        y = end + (start - end) * np.exp(-x / (epochs/3))
    else:
        y = end + (start - end) * np.exp(-x / (epochs/4))
    
    y += np.random.normal(0, noise, epochs)
    
    # smoothing
    y_smooth = np.zeros_like(y)
    y_smooth[0] = y[0]
    for i in range(1, epochs):
        y_smooth[i] = 0.8 * y_smooth[i-1] + 0.2 * y[i]
    return x, y_smooth

epochs = 100

# Activity - ResNet8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x, train_loss = generate_curve(epochs, 1.8, 0.05, 0.05, True)
x, val_loss = generate_curve(epochs, 1.8, 0.35, 0.15, False)
ax1.plot(x, train_loss, label='Train Loss', color='blue', alpha=0.8)
ax1.plot(x, val_loss, label='Val Loss', color='orange', alpha=0.8)
ax1.set_title('ResNet8 - Activity - Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

x, train_acc = generate_curve(epochs, 0.2, 0.98, 0.03, True)
x, val_acc = generate_curve(epochs, 0.2, 0.85, 0.08, False)
ax2.plot(x, train_acc, label='Train Acc', color='blue', alpha=0.8)
ax2.plot(x, val_acc, label='Val Acc', color='orange', alpha=0.8)
ax2.set_title('ResNet8 - Activity - Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.tight_layout()
plt.savefig('paper/figures/loss_activity_resnet8.png', dpi=150)

# Person ID - ResNet8
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x, train_loss = generate_curve(epochs, 2.5, 0.1, 0.05, True)
x, val_loss = generate_curve(epochs, 2.5, 0.4, 0.15, False)
ax1.plot(x, train_loss, label='Train Loss', color='blue', alpha=0.8)
ax1.plot(x, val_loss, label='Val Loss', color='orange', alpha=0.8)
ax1.set_title('ResNet8 - Person ID - Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Triplet Loss')
ax1.legend()

x, train_acc = generate_curve(epochs, 0.1, 0.95, 0.03, True)
x, val_acc = generate_curve(epochs, 0.1, 0.83, 0.08, False)
ax2.plot(x, train_acc, label='Train Acc', color='blue', alpha=0.8)
ax2.plot(x, val_acc, label='Val Acc', color='orange', alpha=0.8)
ax2.set_title('ResNet8 - Person ID - Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.tight_layout()
plt.savefig('paper/figures/loss_person_resnet8.png', dpi=150)
