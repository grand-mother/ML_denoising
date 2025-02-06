import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from  traces_visualization import prepare_denoise_data
import os
#running python script properly
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define the custom dataset class
class NoisyDataset(Dataset):
    """A dataset that adds noise to the original data."""

    def __init__(self, original_dataset, noise_factor=0.5):
        self.original_dataset = original_dataset
        self.noise_factor = noise_factor

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        pure_data = self.original_dataset[idx]
        noise_data = pure_data + torch.randn_like(pure_data) * self.noise_factor
        noise_data = torch.clamp(noise_data, -1., 1.)
        return pure_data, noise_data


# Define the Autoencoder model class
class Autoencoder(nn.Module):
    """A simple Autoencoder model."""

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool1d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def initialize_model(learning_rate=0.01):
    """Initialize the model and optimizer."""
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


# Preparation for training
train_loader, valid_loader, test_loader = prepare_denoise_data(ndat_train=1024, ndat_valid=128, ndat_test=5,batch_size=16)
noisy_train_loader = DataLoader(NoisyDataset(train_loader.dataset), batch_size=16, shuffle=True)
noisy_test_loader = DataLoader(NoisyDataset(test_loader.dataset), batch_size=16, shuffle=True)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
print(device)

model, optimizer = initialize_model()
criterion = nn.MSELoss()

train_losses = []
test_losses = []
num_epochs = 100

# Training Phase
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for pure_data, noise_data in noisy_train_loader:
        pure_data, noise_data = pure_data.to(device), noise_data.to(device)
        optimizer.zero_grad()
        outputs = model(noise_data)
        loss = criterion(outputs, pure_data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(noisy_train_loader))

    # Evaluation Phase
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for pure_data, noise_data in noisy_test_loader:
            pure_data, noise_data = pure_data.to(device), noise_data.to(device)
            outputs = model(noise_data)
            loss = criterion(outputs, pure_data)
            test_loss += loss.item()
    test_losses.append(test_loss / len(noisy_test_loader))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss over Epochs')
plt.legend()
plt.show()

# Testing the denoising capability
model.eval()
originals, noisy_data, denoised_data = [], [], []
with torch.no_grad():
    for pure_data, noise_data in noisy_test_loader:
        pure_data, noise_data = pure_data.to(device), noise_data.to(device)
        outputs = model(noise_data)
        for i in range(pure_data.shape[0]):
            originals.append(pure_data[i].cpu().numpy())
            noisy_data.append(noise_data[i].cpu().numpy())
            denoised_data.append(outputs[i].cpu().numpy())

# Visualize original vs noisy vs denoised for a few examples
num_examples = len(noisy_data)
for i in range(num_examples):
    # Plot Pure and Denoised data on one plot
    fig, axs = plt.subplots(2, sharex=True, sharey=True, figsize=(16,9))
    axs[0].plot(originals[i][0, :], label='Pure', color = 'blue')
    axs[0].set_title(f'Example {i + 1}: Pure and Denoised Data')
    axs[0].legend(loc='upper right')
    axs[0].plot(denoised_data[i][0, :], label='Denoised', linestyle='--', color = 'orange')
    axs[0].legend(loc='upper right')

    # Plot the noise on the another plot
    axs[1].plot(noisy_data[i][0, :], color='red', label='Noise')
    axs[1].set_title(f'Example {i + 1}: Noised Data')
    axs[1].legend(loc='upper right')
    for axs in axs.flat:
      axs.set(xlabel='time', ylabel='Amplitude')
    plt.show()  # Show plots

print('Simulation is complete')
