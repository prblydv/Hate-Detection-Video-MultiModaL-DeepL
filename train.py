

# this is the best as it accumulates the graident for each n number of batched

import torch
from torch import nn, optim
from tqdm import tqdm
from data_provider.data_factory import data_getter
from hatemod import MultiChannelModel  # Your model
from utils import adjust_learning_rate

# Enable CuDNN auto-tuning for better performance
torch.backends.cudnn.benchmark = True

# Gradient Accumulation Steps
# ACCUMULATION_STEPS = 4  # Adjust this based on GPU memory

def train_model(config):
    """
    Train the MultiChannelModel using the provided configuration.
    :param config: Configuration dictionary for data and training parameters.
    :return: Lists of train_losses, val_losses, train_accuracies, val_accuracies
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ACCUMULATION_STEPS = config['accF']  # Adjust this based on GPU memory
    # Create DataLoaders
    train_loader = data_getter({**config, 'flag': 'train'})
    val_loader = data_getter({**config, 'flag': 'val'})

    # Initialize model, loss function, and optimizer
    model = MultiChannelModel().to(device)
    criterion = nn.BCELoss()
    # criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler()

    epochs = config.get('epochs', 10)
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        adjust_learning_rate(optimizer, epoch, config)  # Adjust LR if necessary

        # Training Phase
        model.train()
        train_loss, correct, total = 0, 0, 0

        print(f"Epoch {epoch}/{epochs}")

        optimizer.zero_grad()  # Clear gradients before accumulation

        # CUDA Stream for parallel execution
        stream = torch.cuda.Stream()

        for batch_idx, (frames, audio, transcript, labels) in enumerate(tqdm(train_loader)):
            frames, audio, transcript, labels = (
                frames.to(device, non_blocking=True),
                audio.to(device, non_blocking=True),
                transcript.to(device, non_blocking=True),
                labels.float().to(device, non_blocking=True)
            )

            with torch.cuda.stream(stream):  # Enable CUDA stream for parallel execution
                with torch.cuda.amp.autocast():  # Mixed precision training
                    outputs = model(audio, frames, transcript)
                with torch.cuda.amp.autocast(enabled=False):  
                    loss = criterion(outputs.squeeze().float(), labels.float()) / ACCUMULATION_STEPS  # Ensure loss is float32

                # loss = criterion(outputs.squeeze(), labels.float()) / ACCUMULATION_STEPS  # Normalize loss

                scaler.scale(loss).backward()  # Scale loss before backward pass

                # Perform optimizer step only every ACCUMULATION_STEPS batches
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()  # Reset gradients after step

                train_loss += loss.item() * ACCUMULATION_STEPS  # Reverse normalization for reporting
                preds = (outputs.squeeze() > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        train_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for frames, audio, transcript, labels in val_loader:
                frames, audio, transcript, labels = (
                    frames.to(device),
                    audio.to(device),
                    transcript.to(device),
                    labels.to(device)
                )

                outputs = model(audio, frames, transcript)
                # print(outputs)
                loss = criterion(outputs.squeeze(), labels.float())

                val_loss += loss.item()
                preds = (outputs.squeeze() > 0.5).float()
                # print(preds,labels)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        # print(val_correct)
        val_acc = val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f"Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_acc:.4f}")

        torch.cuda.empty_cache()

    return train_losses, val_losses, train_accuracies, val_accuracies













