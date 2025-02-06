




















import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomResNetVideo(nn.Module):
    def __init__(self):
        super(CustomResNetVideo, self).__init__()
        # Convolutional layer to process the frames
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.relu = nn.ReLU()
        nn.Dropout3d(p=0.3),  # Apply Spatial Dropout for 3D Conv layers

        self.pool3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Another convolutional layer
        self.conv3d_2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        # nn.Dropout3d(p=0.3),  # Apply Spatial Dropout for 3D Conv layers
        
        # Fully connected layers (initialized later)
        self.fc1 = None
        self.fc2 = nn.Linear(256, 128)  # Output layer with 1 neuron for binary classification

    def forward(self, x):
        # x has shape [batch_size, num_frames, channels, height, width]
        x = x.permute(0, 2, 1, 3, 4)  # Change shape to [batch_size, channels, num_frames, height, width]

        # Pass through the first 3D convolutional layer
        x = self.conv3d(x)
        x = self.relu(x)
        x = self.pool3d(x)  # Adjusted feature size for 64x64 input

        # Pass through the second 3D convolutional layer
        x = self.conv3d_2(x)
        x = self.relu(x)
        x = self.pool3d(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Dynamically initialize fc1 based on the input size
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Shape: [batch_size, 1]

        return x


# # Example usage
# if __name__ == "__main__":
#     # Dummy input tensor with shape [batch_size, num_frames, channels, height, width]
#     batch_size = 16
#     num_frames = 50
#     channels = 3
#     height, width = 64, 64  # Updated input dimensions
#     input_tensor = torch.randn(batch_size, num_frames, channels, height, width)

#     # Initialize model and pass the input
#     model = CustomResNetVideo()
#     output = model(input_tensor)
#     print("Output shape:", output.shape)  # Expected output: [batch_size, 1]


































































