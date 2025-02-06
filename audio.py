# audio for preprocesse video
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 3 channels for RGB spectrogram
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Pooling Layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 32 * 32, 512)  # Flattened feature size
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 128)  # Final layer for binary classification
        # self.fc4 = nn.Linear(4,1)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional Layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 128 * 32 * 32)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # print('Passed the Tensors from audio.py file with fc2 shape:',x.shape )

        x = self.dropout(x)
        x = self.fc3(x)  # Final output layer
        # print('Passed the Tensors from audio.py file with fc3 shape:',x.shape )
        return x  # Sigmoid activation for binary classification
        
        # return torch.sigmoid(x)  # Sigmoid activation for binary classification
