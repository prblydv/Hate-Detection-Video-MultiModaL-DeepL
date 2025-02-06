# old datloader suign preprocessed vidoes

import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class HateMMDataLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset loader for hate speech multimedia data.

        :param root_dir: Root directory containing processed videos.
        :param transform: Optional torchvision transforms for preprocessing images.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Dynamically load video folders and determine labels
        video_folders = [folder for folder in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, folder))]
        self.data = []
        
        for folder in video_folders:
            label = 1 if folder.startswith('hate_video') else 0  # Hate=1, Non-hate=0
            folder_path = os.path.join(root_dir, folder)
            self.data.append((folder_path, label))
    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches a single data point by index.

        :param idx: Index of the data point to fetch.
        :return: A tuple containing frames, audio spectrogram, tokenized transcript tensor, label, and metadata.
        """
        # print('get item is called with idx:', idx)
        # Get video metadata
        video_folder, label = self.data[idx]
        frames_folder = os.path.join(video_folder, "frames")

        # Check if the frames folder exists
        if not os.path.exists(frames_folder):
            raise FileNotFoundError(f"Frames folder not found: {frames_folder}")

        # Process data
        frames = self._load_frames(frames_folder)

        # Audio spectrogram processing
        audio_path = os.path.join(video_folder, "audio.png")
        audio_data = self._load_audio(audio_path)

        # Tokenized transcript
        transcript_path = os.path.join(video_folder, "transcript.txt")
        transcript = self._load_transcript(transcript_path)  # Returns a tensor


        return frames, audio_data, transcript, label

    def _load_frames(self, frames_folder):
        """
        Loads all files in the frames folder, applies necessary transformations, and ensures the number of frames is 50.
        If there are fewer than 50 frames, black frames are added to make up the difference.
        """
        frame_files = sorted(os.listdir(frames_folder))  # Ensure sorted order
        frames = []

        # Load all frames from the folder
        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            image = Image.open(frame_path).convert("RGB")
            if self.transform:  # Apply transformations if specified
                image = self.transform(image)
            frames.append(image)

        # Ensure the number of frames is 50
        num_frames = len(frames)
        if num_frames < 50:
            # Get the size of existing frames (assume all frames have the same size)
            black_frame = torch.zeros_like(frames[0]) if num_frames > 0 else None
            while len(frames) < 50:
                frames.append(black_frame)

        # Convert the list of tensors to a single tensor
        return torch.stack(frames[:50])  # Truncate to 50 if there are more than 50 frames

    def _load_audio(self, audio_path):
        """
        Loads the audio spectrogram from the 'audio.png' file.
        """
        if os.path.exists(audio_path):
            image = Image.open(audio_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
        else:
            return torch.zeros(3, 224, 224)  # Default zero tensor if the audio spectrogram is missing

    def _load_transcript(self, transcript_path):
        """
        Loads and preprocesses the transcript as tokenized input for the text model.

        :param transcript_path: Path to the transcript text file.
        :return: Concatenated tensor combining input_ids and attention_mask.
        """
        # Default tensors for missing transcripts
        default_input_ids = torch.zeros(512, dtype=torch.long)  # 512 is max length
        default_attention_mask = torch.zeros(512, dtype=torch.long)

        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as file:
                text = file.read().strip()

            # Tokenize the text
            encoded = tokenizer(
                text,
                padding="max_length",  # Pad to max length for uniformity
                truncation=True,       # Truncate if the text exceeds max length
                max_length=512,        # Match the max length of the model
                return_tensors="pt"    # Return as PyTorch tensors
            )

            input_ids = encoded['input_ids'].squeeze(0)  # Remove batch dimension
            attention_mask = encoded['attention_mask'].squeeze(0)

            return torch.cat((input_ids, attention_mask), dim=0)
        else:
            # Return concatenated default tensors if the transcript is missing
            return torch.cat((default_input_ids, default_attention_mask), dim=0)














