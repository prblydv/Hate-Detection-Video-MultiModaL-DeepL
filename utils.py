import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch, config):
    """
    Adjust the learning rate based on the specified strategy in the config.
    :param optimizer: Optimizer object.
    :param epoch: Current epoch.
    :param config: Configuration dictionary with learning rate adjustment parameters.
    """
    lr_adjust = {}
    lr = config.get('lr', 1e-3)
    if config['lradj'] == 'type1':
        lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    elif config['lradj'] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif config['lradj'] == 'type3':
        lr_adjust = {epoch: lr}
    elif config['lradj'] == 'type4':
        lr_adjust = {epoch: lr * (0.9 ** ((epoch - 1) // 1))}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots the training and validation loss/accuracy.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))

    # Loss plot
    axs[0].plot(epochs, train_losses, label="train_loss")
    axs[0].plot(epochs, val_losses, label="valid_loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid()

    # Accuracy plot
    axs[1].plot(epochs, train_accuracies, label="train_acc")
    axs[1].plot(epochs, val_accuracies, label="valid_acc")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("ACC")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()