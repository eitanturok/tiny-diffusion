import torch
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera


def save_model(model, outdir):
    model_path = f"{outdir}/model.pth"
    print(f"Saving model in {model_path}...")
    torch.save(model.state_dict(), model_path)

def save_losses(losses, outdir):
    loss_path = f"{outdir}/loss.npy"
    print(f"Saving loss as numpy array in {loss_path}...")
    np.save(loss_path, np.array(losses))

def save_frames(frames, outdir):
    frame_path = f"{outdir}/frames.npy"
    print(f"Saving frames in {frame_path}...")
    np.save(frame_path, frames)

def animate_training(frames, outdir):
    train_path = f"{outdir}/train.mp4"
    print(f"Animating training in {train_path}...")

    xmin, xmax = -3.5, 3.5
    ymin, ymax = -4., 4.75
    fig, ax = plt.subplots()
    camera = Camera(fig)

    # reverse process
    for i, sample in enumerate(frames):
        plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=15, color="blue")
        ax.text(0.0, 0.95, f"Train step {i: 4} / {len(frames)-1}", transform=ax.transAxes)
        ax.text(0.0, 1.01, "Sample Algorithm during training", transform=ax.transAxes, size=15)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.axis("off")
        camera.snap()

    animation = camera.animate(blit=True, interval=35)
    animation.save(train_path)
    return animation

def animate_reverse_process(reverse_samples, num_timesteps, outdir):

    reverse_path = f"{outdir}/reverse_process.mp4"
    print(f"Animating reverse process in {reverse_path}...")

    xmin, xmax = -3.5, 3.5
    ymin, ymax = -4., 4.75
    fig, ax = plt.subplots()
    camera = Camera(fig)

    # reverse process
    for i, sample in enumerate(reverse_samples):
        plt.scatter(sample[:, 0], sample[:, 1], alpha=0.5, s=15, color="blue")
        ax.text(0.0, 0.95, f"Sample step {i: 4} / {num_timesteps}", transform=ax.transAxes)
        ax.text(0.0, 1.01, "Reverse Process after training", transform=ax.transAxes, size=15)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.axis("off")
        camera.snap()

    animation = camera.animate(blit=True, interval=35)
    animation.save(reverse_path)
    return animation
