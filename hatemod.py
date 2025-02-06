import torch
import torch.nn as nn
import torch.nn.functional as F
from audio import AudioClassifier
from video import CustomResNetVideo
from text import TextClassifier


class MultiChannelModel(nn.Module):
    def __init__(self, 
                 audio_dim=(256, 256, 3), 
                 video_dim=(224, 224, 3), 
                 num_frames=50, 
                 text_hidden_size=768, 
                 num_classes=1, 
                 num_heads=4, 
                 dropout=0.2):
        """
        Multi-modal model combining audio, video, and text features.
        
        :param audio_dim: Shape of the input spectrogram (Height, Width, Channels)
        :param video_dim: Shape of video input (Height, Width, Channels)
        :param num_frames: Number of frames in each video sample.
        :param text_hidden_size: Size of the BERT-based text feature vector.
        :param num_classes: Number of output classes (default=1 for binary classification).
        :param num_heads: Number of attention heads for multi-head attention.
        :param dropout: Dropout rate for regularization.
        """
        super(MultiChannelModel, self).__init__()

        self.audio_height, self.audio_width, self.audio_channels = audio_dim
        self.video_height, self.video_width, self.video_channels = video_dim
        self.num_frames = num_frames

        # Initialize individual models
        self.audio_model = AudioClassifier()
        self.video_model = CustomResNetVideo()
        self.text_model = TextClassifier()

        # Define input dimensions
        self.in_dim = 128 * 3  # Assuming each model outputs a 128-dim vector
        self.attention_dim = 128 * 3  # Attention processing dimension

        # Multi-head Self-Attention for Feature Fusion
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.attention_dim, 
                                                         num_heads=num_heads, 
                                                         dropout=dropout, 
                                                         batch_first=True)

        # Fully connected layers with dropout and GELU activation
        self.fc_layers = nn.Sequential(
            nn.Linear(self.attention_dim, self.attention_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.attention_dim // 2, self.attention_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.attention_dim // 4, num_classes)
        )
        self.sigmoid = nn.Sigmoid()

        # Apply weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming Initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, audio_input, video_input, text_input):
        """
        Forward pass for the multi-modal model.
        :param audio_input: Spectrogram input for the audio channel (batch_size, channels, height, width).
        :param video_input: Frames input for the video channel (batch_size, num_frames, channels, height, width).
        :param text_input: Text input for the text channel (batch_size, variable_length_text).
        :return: Prediction logits (batch_size, num_classes).
        """
        # Ensure inputs are on the correct device
        device = next(self.parameters()).device
        audio_input, video_input, text_input = audio_input.to(device), video_input.to(device), text_input.to(device)

        # Extract features from each modality
        video_output = self.video_model(video_input)  # (batch_size, feature_dim)
        audio_output = self.audio_model(audio_input)  # (batch_size, feature_dim)
        text_output = self.text_model(text_input)     # (batch_size, feature_dim)

        # Concatenate extracted features
        combined_output = torch.cat((audio_output, video_output, text_output), dim=1).unsqueeze(1)  # Shape: (batch, seq_len=1, features)

        # Apply Multi-head Self-Attention
        attention_output, _ = self.multihead_attention(combined_output, combined_output, combined_output)

        # Remove sequence dimension and pass through fully connected layers
        output = self.fc_layers(attention_output.squeeze(1))  # (batch_size, num_classes)
        output = self.sigmoid(output)

        return output



























# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from audio import AudioClassifier
# from video import CustomResNetVideo
# from text import TextClassifier

# class MultiChannelModel(nn.Module):
#     def __init__(self, audio_dim=(256, 256, 3), video_dim=(224, 224, 3), num_frames=50, text_hidden_size=768, num_classes=4):
#         """
#         Multi-channel model combining audio, video, and text models.
#         :param audio_dim: Tuple of (height, width, channels) for audio input.
#         :param video_dim: Tuple of (height, width, channels) for video frames.
#         :param num_frames: Number of frames in each video sample.
#         :param text_hidden_size: The hidden size output from the BERT model.
#         :param num_classes: The number of output classes.
#         """
#         super(MultiChannelModel, self).__init__()

#         self.audio_height, self.audio_width, self.audio_channels = audio_dim
#         self.video_height, self.video_width, self.video_channels = video_dim
#         self.num_frames = num_frames

#         # Initialize individual models
#         self.audio_model = AudioClassifier()  # Audio Classifier
#         self.video_model = CustomResNetVideo()  # Video Classifier
#         self.text_model = TextClassifier()  # Text Classifier

#         # Define input and attention dimensions
#         self.in_dim = 64 * 3
#         self.attention_dim = 64 * 4
#         self.fc = nn.Linear(self.attention_dim, num_classes)

#         # Attention weight matrices
#         self.W_q = nn.Linear(self.in_dim, self.attention_dim, bias=False)  # [N → d]
#         self.W_k = nn.Linear(self.in_dim, self.attention_dim, bias=False)  # [N → d]
#         self.W_v = nn.Linear(self.in_dim, self.attention_dim, bias=False)  # [N → d]

#         # Scaling factor for attention
#         self.scale = torch.sqrt(torch.FloatTensor([self.attention_dim]))

#     def forward(self, audio_input, video_input, text_input):
#         """
#         Forward pass for the multi-channel model.
#         :param audio_input: Spectrogram input for the audio channel (batch_size, channels, height, width).
#         :param video_input: Frames input for the video channel (batch_size, num_frames, channels, height, width).
#         :param text_input: Text input for the text channel (batch_size, variable_length_text).
#         :return: Softmax probabilities (batch_size, num_classes).
#         """
#         video_output = self.video_model(video_input)  # Shape: (batch_size, video_flatten_dim)
#         audio_output = self.audio_model(audio_input)  # Shape: (batch_size, audio_flatten_dim)
#         text_output = self.text_model(text_input)     # Shape: (batch_size, text_hidden_size)

#         # Concatenate outputs from all channels
#         combined_output = torch.cat((audio_output, video_output, text_output), dim=1)  # Shape: (batch_size, combined_dim)

#         # Apply Attention Mechanism
#         Q = self.W_q(combined_output)  # Shape: [B, M, d]
#         K = self.W_k(combined_output)  # Shape: [B, M, d]
#         V = self.W_v(combined_output)  # Shape: [B, M, d]

#         # Compute attention scores (QK^T) and scale
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, M, M]

#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention_scores, dim=-1)  # [B, M, M]

#         # Multiply attention weights with V
#         attention_output = torch.matmul(attention_weights, V)  # [B, M, d]

#         # Fully connected layer
#         logits = self.fc(attention_output)  

#         # Apply softmax for multi-class classification
#         probabilities = F.softmax(logits, dim=1)  

#         return probabilities


















# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from audio import AudioClassifier
# from video import CustomResNetVideo
# from text import TextClassifier

# class MultiChannelModel(nn.Module):
#     def __init__(self, audio_dim=(256, 256, 3), video_dim=(224, 224, 3), num_frames=50, text_hidden_size=768, num_classes=4):
#         """
#         Multi-channel model combining audio, video, and text models.
#         :param audio_dim: Tuple of (height, width, channels) for audio input.
#         :param video_dim: Tuple of (height, width, channels) for video frames.
#         :param num_frames: Number of frames in each video sample.
#         :param text_hidden_size: The hidden size output from the BERT model.
#         :param num_classes: The number of output classes.
#         """
#         super(MultiChannelModel, self).__init__()

#         self.audio_height, self.audio_width, self.audio_channels = audio_dim
#         self.video_height, self.video_width, self.video_channels = video_dim
#         self.num_frames = num_frames
#         # Initialize individual models
#         self.audio_model = AudioClassifier()  # Audio Classifier
#         self.video_model = CustomResNetVideo()  # Video Classifier
#         self.text_model = TextClassifier()  # Text Classifier

#         # Define input and attention dimensions
#         self.in_dim = 128 * 3
#         self.attention_dim = 128 * 4

#         # Attention weight matrices
#         self.W_q = nn.Linear(self.in_dim, self.attention_dim, bias=False)  # [N → d]
#         self.W_k = nn.Linear(self.in_dim, self.attention_dim, bias=False)  # [N → d]
#         self.W_v = nn.Linear(self.in_dim, self.attention_dim, bias=False)  # [N → d]

#         # Scaling factor for attention
#         self.scale = torch.sqrt(torch.FloatTensor([self.attention_dim]))

#         # Fully connected layers
#         self.fc = nn.Sequential(
#             nn.Linear(self.attention_dim, self.attention_dim//2),  # First FC layer (reduces to 2)
#             nn.GELU(),
#             nn.Linear(self.attention_dim//2, self.attention_dim//4),
#             nn.GELU(),
#             nn.Linear(self.attention_dim//4, self.attention_dim//8),
#             nn.GELU(),
#             nn.Linear(self.attention_dim//8, 1)
#         )


#         # self.sf = nn.Sigmoid()
#     def forward(self, audio_input, video_input, text_input):
#         """
#         Forward pass for the multi-channel model.
#         """

#         # Ensure all inputs are on the same device as the model
#         device = next(self.parameters()).device
#         audio_input = audio_input.to(device)
#         video_input = video_input.to(device)
#         text_input = text_input.to(device)

#         # Pass inputs through individual models
#         video_output = self.video_model(video_input)  
#         audio_output = self.audio_model(audio_input)  
#         text_output = self.text_model(text_input)  

#         # Concatenate outputs
#         combined_output = torch.cat((audio_output, video_output, text_output), dim=1).to(device)

#         # Apply attention mechanism (Ensure tensors are on the same device)
#         Q = self.W_q(combined_output).to(device)  
#         K = self.W_k(combined_output).to(device)  
#         V = self.W_v(combined_output).to(device)  

#         # Compute attention scores (QK^T) and scale
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(device)

#         # Apply softmax to get attention weights
#         attention_weights = F.softmax(attention_scores, dim=-1)

#         # Multiply attention weights with V
#         attention_output = torch.matmul(attention_weights, V)

#         # Final fully connected layer
#         output = self.fc(attention_output)
#         # output= self.sf(output)

#         return output
