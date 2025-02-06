import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch


class HateMMDataLoader(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Initializes the dataset loader for hate speech multimedia data.
        
        :param root_dir: Root directory containing processed videos.
        :param annotation_file: Path to the CSV file containing annotations.
        :param transform: Optional torchvision transforms for preprocessing images.
        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        """
        Returns the size of the dataset.
        Changed: Returns the number of folders in the root directory instead of the number of rows in the annotation file.
        """
        return len([folder for folder in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, folder))])

    def __getitem__(self, idx):
        """
        Fetches a single data point by index.
        :param idx: Index of the data point to fetch.
        :return: A tuple containing frames, audio spectrogram, transcript, label, and metadata.
        """
        video_name = self.annotations.iloc[idx, 0]
        label = 1 if video_name.startswith('hate_video') else 0  # Label: hate_video=1, non_hate_video=0
        hate_target = self.annotations.iloc[idx, 3]

        # Frame processing
        video_folder = os.path.join(self.root_dir, video_name.split('.')[0])
        frames_folder = os.path.join(video_folder, "frames")
        frames = self._load_frames(frames_folder)

        # Audio spectrogram processing
        audio_path = os.path.join(video_folder, "audio.png")
        audio_data = self._load_audio(audio_path)

        # Transcript
        transcript_path = os.path.join(video_folder, "transcript.txt")
        transcript = self._load_transcript(transcript_path)

        metadata = {
            "target": hate_target
        }

        return frames, audio_data, transcript, label, metadata

    def _load_frames(self, frames_folder):
        """
        Loads all files in the frames folder without any processing.
        Changed: Returns all frames as-is without applying any transformations or limiting the number of frames.
        """
        frame_files = sorted(os.listdir(frames_folder))  # Ensure sorted order
        frames = []
        for frame_file in frame_files:
            frame_path = os.path.join(frames_folder, frame_file)
            image = Image.open(frame_path).convert("RGB")
            frames.append(image)
        return frames  # List of PIL images

    def _load_audio(self, audio_path):
        """
        Loads the audio spectrogram from the 'audio.png' file.
        Changed: Directly loads the spectrogram image from the provided path.
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
        Loads the transcript as a string.
        Changed: Only loads the transcript if the file exists, else returns an empty string.
        """
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as file:
                return file.read().strip()
        else:
            return ""


# Example Usage
if __name__ == "__main__":
    # Dataset root and annotation file
    dataset_root = "path/to/ProcessedVideos"  # Replace with your dataset root
    annotation_csv = "path/to/HateMM_annotation.csv"  # Replace with your annotation CSV path

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Initialize dataset
    dataset = HateMMDataLoader(
        root_dir=dataset_root,
        annotation_file=annotation_csv,
        transform=transform
    )

    # Example DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Iterate through the DataLoader
    for i, (frames, audio, transcript, label, metadata) in enumerate(dataloader):
        print(f"Batch {i + 1}")
        print(f"Number of Frames: {len(frames[0])}")  # Number of frames in the first video
        print(f"Audio Spectrogram Shape: {audio.shape}")  # Shape of the audio spectrogram
        print(f"Labels: {label}")  # Shape: (batch_size,)
        print(f"Metadata: {metadata}")  # Dictionary with 'target'
        break



# Example Usage
if __name__ == "__main__":
    # Dataset root and annotation file
    dataset_root = "path/to/ProcessedVideos"  # Replace with your dataset root
    annotation_csv = "path/to/HateMM_annotation.csv"  # Replace with your annotation CSV path

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Configuration
    config = {
        "num_frames": 50,
        "audio_shape": (256, 256),
        "batch_size": 8,
        "num_workers": 4,
        "drop_last": True
    }

    # Initialize dataset
    dataset = HateMMDataLoader(
        root_dir=dataset_root, 
        annotation_file=annotation_csv, 
        transform=transform, 
        config=config
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,  # Shuffle only for training
        num_workers=config['num_workers'],
        drop_last=config['drop_last']  # Use drop_last from config
    )

    # Iterate through the DataLoader
    for i, (frames, audio, transcript, label, metadata) in enumerate(dataloader):
        print(f"Batch {i + 1}")
        print(f"Frames shape: {frames.shape}")  # Shape: (batch_size, num_frames, channels, height, width)
        print(f"Audio shape: {audio.shape}")    # Shape: (batch_size, 256, 256)
        print(f"Labels: {label}")              # Shape: (batch_size,)
        print(f"Metadata: {metadata}")         # Dictionary with 'target'
        break
