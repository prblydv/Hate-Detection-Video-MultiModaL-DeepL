
# old data factory using preporcessedvideo


import os
from torch.utils.data import DataLoader
from torchvision import transforms
from .data_loader import HateMMDataLoader
import random

def data_getter(config):
    """
    A factory function to create DataLoader instances for different dataset flags.

    :param config: Configuration dictionary containing all necessary parameters.
                   Example:
                   {
                       'root_dir': './ProcessedVideos',
                       'flag': 'train',  # One of ['train', 'val', 'test']
                       'batch_size': 8,
                       'num_workers': 4
                   }
    :return: DataLoader instance.
    """
    # Default values
    default_config = {
        'root_dir': './ProcessedVideos',
        'flag': 'train',
        'batch_size': 8,
        'num_workers': os.cpu_count()  # Default to the number of available CPU cores
    }
    
    # Update defaults with provided config
    for key, value in default_config.items():
        config.setdefault(key, value)

    # Validate flag
    assert config['flag'] in ['train', 'test', 'val'], "Flag must be one of ['train', 'test', 'val']"

    # Get all video folders
    video_folders = os.listdir(config['root_dir'])
    video_folders = [folder for folder in video_folders if os.path.isdir(os.path.join(config['root_dir'], folder))]

    # Shuffle the folders to ensure randomness
    random.seed(42)  # For reproducibility
    random.shuffle(video_folders)

    # Determine split sizes
    total_videos = len(video_folders)
    train_size = int(0.8 * total_videos)
    val_size = int(0.2 * total_videos)

    if config['flag'] == 'train':
        selected_folders = video_folders[:train_size]
    elif config['flag'] == 'val':
        selected_folders = video_folders[train_size:train_size + val_size]
    elif config['flag'] == 'test':
        selected_folders = video_folders[train_size + val_size:]

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    # Initialize dataset with selected folders
    dataset = HateMMDataLoader(
        root_dir=config['root_dir'],
        transform=transform
    )
    # Filter dataset based on selected folders
    dataset.data = [(path, label) for path, label in dataset.data if os.path.basename(path) in selected_folders]

    print(f"Total number of videos in {config['flag']} dataset: {len(dataset)}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True if config['flag'] == 'train' else False,
        num_workers=config['num_workers'],  # Use the num_workers from config
        pin_memory=True,  # Speeds up data transfer to GPU
        prefetch_factor=4,  # Number of batches to prefetch per worker
        persistent_workers=True,  # Keeps workers alive for faster loading
    )

    return dataloader


