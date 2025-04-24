# "Denoising Diffusion Probabilistic Models" https://arxiv.org/abs/2006.11239
# Resources:
# Annotated Diffusion https://huggingface.co/blog/annotated-diffusion
import argparse
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

import datasets
from positional_embeddings import PositionalEmbedding

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + \
            len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class NoiseScheduler():
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    # reverse process
    # sample x_{t-1} ~ p_\theta( x_{t-1} | x_t ) in equation (1) right column
    # part of the Sampling Algorithm steps 3-4
    def remove_noise(self, noisy_x:torch.Tensor, t:torch.Tensor, pred_noise:torch.Tensor):
        # Step 3: sample gaussian noise
        noise = torch.randn_like(pred_noise) if t > 0 else torch.zeros_like(pred_noise)

        # Step 4: compute x_prev
        # Equation (11) for mean_prev
        noise_coeff = self.betas[t] / (1 - self.alphas_cumprod[t]).sqrt()
        mean_prev = (1 / self.alphas[t]).sqrt() * (noisy_x - noise_coeff * pred_noise)
        # Equation (7) right column for variance_prev
        variance_prev = torch.zeros(1) if t != 0 else self.betas[t] * (1. - self.alphas_cumprod[t-1]) / (1. - self.alphas_cumprod[t])
        x_prev = mean_prev + variance_prev.sqrt() * noise
        return x_prev

    # forward process
    # sample x_t ~ q( x_t | x_0 ) in equation (4)
    # part of the Training Algorithm, step 5
    def add_noise(self, x_start:torch.Tensor, noise:torch.Tensor, timesteps:torch.Tensor):
        mean_coeff = self.alphas_cumprod[timesteps].sqrt().reshape(-1, 1)
        var_coeff = (1 - self.alphas_cumprod[timesteps]).sqrt().reshape(-1, 1)
        noisy_x = mean_coeff * x_start + var_coeff * noise
        return noisy_x

    def __len__(self):
        return self.num_timesteps

# "Algorithm 1: Training" from paper, pg 4.
def train(batch, model, noise_scheduler:NoiseScheduler, optimizer:AdamW):
    model.train()

    # Step 2: this step is implicitly performed when we choose our data x
    x, batch_size = batch['data'], batch['data'].shape[0]

    # Step 3: uniformly sample a noise level t for each image
    t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=x.device).long() #(b,)
    # Step 4: sample noise from a standard normal
    noise = torch.randn(x.shape) # (b,c,h,w)

    # Step 5 (multiple parts):
    noisy_x = noise_scheduler.add_noise(x, noise, t) # forward process
    pred_noise = model(noisy_x, t)
    loss = F.mse_loss(pred_noise, noise, reduction='mean')
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    ret = {
        't': t,
        'noise': noise,
        'noisy_x': noisy_x,
        'pred_noise': pred_noise,
        'loss': loss.detach().item(),
    }
    return ret

# "Algorithm 2: Sampling" from paper, pg 4.
@torch.no_grad()
def sample(batch_shape, model, noise_scheduler:NoiseScheduler, eval_batch_size:int=1):
    model.eval()
    device = next(model.parameters()).device

    # Step 1: initialize image as standard gaussian noise
    curr_image = torch.randn((eval_batch_size, *batch_shape[1:]), device=device) # (1,c,h,w)
    images, prev_image = [curr_image], None

    # Step 2: denoise image in T steps in decreasing order
    for t_scalar in tqdm(list(range(noise_scheduler.num_timesteps)[::-1])):
        t = torch.full((eval_batch_size,), t_scalar, device=device).long() #(b,)
        # Steps 3, 4 are performed in the noise scheduler
        pred_noise = model(curr_image, t)
        prev_image = noise_scheduler.remove_noise(curr_image, t_scalar, pred_noise)
        images.append(prev_image)
        curr_image = prev_image

    ret = {
        'image': prev_image,
        'images': images
    }
    return ret

def make_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base")
    parser.add_argument("--dataset", type=str, default="dino", choices=["circle", "dino", "line", "moons"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    config = parser.parse_args()
    return config


def save(config, model, frames, losses):
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)

    model_path = f"{outdir}/model.pth"
    print(f"Saving model in {model_path}...")
    torch.save(model.state_dict(), model_path)

    imgdir = f"{outdir}/images"
    print(f"Saving images in {imgdir}...")
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    loss_path = f"{outdir}/loss.npy"
    print(f"Saving loss as numpy array in {loss_path}...")
    np.save(loss_path, np.array(losses))

    frame_path = f"{outdir}/frames.npy"
    print(f"Saving frames in {frame_path}...")
    np.save(frame_path, frames)

def trainer():

    # init objects
    config = make_config()
    dataset = datasets.get_dataset(config.dataset)
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)
    model = MLP(hidden_size=config.hidden_size, hidden_layers=config.hidden_layers, emb_size=config.embedding_size, time_emb=config.time_embedding, input_emb=config.input_embedding)
    noise_scheduler = NoiseScheduler(num_timesteps=config.num_timesteps)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # train ddpm
    global_step, frames, losses = 0, [], []
    print("Training model...")
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            # train step
            train_ret = train(batch, model, noise_scheduler, optimizer)
            progress_bar.update(1)
            logs = {"loss": train_ret['loss'], "step": global_step}
            losses.append(train_ret['loss'])
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # sample step
            sample_ret = sample(batch['data'].shape, model, noise_scheduler, eval_batch_size=config.eval_batch_size)
            frames.append(sample_ret['image'].numpy())

    # save stuff
    save(config, model, frames, losses)


if __name__ == "__main__":
    trainer()
