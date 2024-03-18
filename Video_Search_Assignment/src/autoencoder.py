import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Ensuring output pixel values are between 0 and 1
        )

    def forward(self, x):
       embedding = self.encoder(x)
       reconstruction = self.decoder(embedding)
       return reconstruction, embedding

def train_autoencoder(autoencoder, data_loader, optimizer, criterion, epochs=5, device=torch.device('cpu')):
    autoencoder.train()
    for epoch in range(epochs):
        for data in data_loader:
            inputs = data.to(device)
            optimizer.zero_grad()
            outputs, _ = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


class InMemoryCroppedObjectDataset(Dataset):
    def __init__(self, cropped_objects,labels = None,transform=None):
        """
        Args:
            cropped_objects (list): List of tensors representing cropped objects.
            transform (callable, optional): Transform to be applied on each tensor.
        """
        self.cropped_objects = cropped_objects
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.cropped_objects)

    def __getitem__(self, idx):
        image = self.cropped_objects[idx]
        label =self.labels[idx] if self.labels is not None else -1
        if self.transform:
            image = self.transform(image)
        return image, label


