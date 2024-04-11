import numpy as np
import matplotlib.pyplot as plt
import torch

from utils.GetLR import get_lr

def plot_loss(log_path,
              warmup_iters, 
              lr_decay_iters, 
              max_lr, 
              min_lr):
    
    # parse logs
    logs = []
    with open(log_path, 'r') as f:
        logs = f.readlines()
        logs = [l.strip() for l in logs][1:]
    logs = [l.split(',') for l in logs]
    logs = np.array(logs).astype(float)
    iterations, train_loss, val_loss = logs.T
    iterations = iterations.astype(int)

    iters = np.arange(0, iterations.max(), 100)
    lrs = [get_lr(it, warmup_iters, lr_decay_iters, max_lr, min_lr) for it in iters]

    # get best model
    best_iter = iterations[np.argmin(val_loss)]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))

    # losses
    ax.semilogy(iterations, train_loss, '-', color='tab:blue', label='Train')
    ax.semilogy(iterations, val_loss, '--', color='tab:blue', label='Val')
    ax.semilogy(best_iter, min(val_loss), '*', color='tab:red', label='Best', markersize=10)
    ax.set_ylabel('Loss', color='tab:blue')
    ax.set_xlabel('Iterations')
    ymin, ymax = ax.get_ylim()
    ymin = 10 ** int(np.log10(ymin) - 1)
    ymax = 10 ** int(np.log10(ymax) + 1)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.legend(loc='upper left')

    # learning rate
    ax2 = ax.twinx()
    ax2.plot(iters, lrs, '-', color='tab:orange', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Learning Rate', color='tab:orange')
    ax2.legend(loc='upper right')

    plt.show()
    
    return None

def plot_val_images(val_loader, model, device, num_plot = 4):
    # options``
    rand_idx = np.random.randint(len(val_loader), size = num_plot)


    # get random validation batch
    (xb, yb) = next(iter(val_loader))
    xb, yb = xb.to(device), yb.to(device)[:, :-1, :, :] # remove weight mask

    # make prediction with unet
    yb_pred = model(xb).to(device)
    yb_pred = torch.nn.functional.softmax(yb_pred, dim=1)

    # convert to cpu, numpy, and change channels for visualization
    xb_numpy = xb.detach().cpu().permute(0, 2, 3, 1).numpy()
    yb_numpy = yb.detach().cpu().permute(0, 2, 3, 1).numpy()[:, :, :, 1:]
    yb_pred_numpy = yb_pred.detach().cpu().permute(0, 2, 3, 1).numpy()

    # convert predicted probabilities to predicted classes and retain four channels
    yb_pred_numpy = np.argmax(yb_pred_numpy, axis=3)
    preds = np.zeros(yb_pred_numpy.shape + (4,))
    for i in range(4):
        preds[:, :, :, i] = yb_pred_numpy == i
    yb_pred_numpy = preds[:, : , :, 1:]

    # yb_pred_numpy = np.concatenate([yb_pred_numpy[:, :, :, None] == i for i in range(4)], axis = 3).astype(int)
    # yb_pred_numpy = yb_pred_numpy[:, :, :, 1:]

    # plot images and masks
    for i in range(num_plot):
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].imshow(xb_numpy[i]); ax[0].set_title('Input')
        ax[1].imshow(yb_numpy[i]); ax[1].set_title('Target')
        ax[2].imshow(yb_pred_numpy[i]); ax[2].set_title('Prediction')
        plt.show()

    return None