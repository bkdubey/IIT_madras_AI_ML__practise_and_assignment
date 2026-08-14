import argparse
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class Generator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # input is latent vector Z
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            # reshape to (128,7,7)
            View((-1, 128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # -> (1,28,28)
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # -> (64,14,14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128,7,7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, 128 * 7 * 7)),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(args):
    # Select device: prefer CUDA, then MPS (Apple), else CPU
    if args.force_cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            # Check for Apple MPS support
            try:
                mps_available = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
            except Exception:
                mps_available = False
            if mps_available:
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # scale to [-1,1]
    ])

    dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    gen = Generator(latent_dim=args.latent_dim).to(device)
    dis = Discriminator().to(device)
    gen.apply(weights_init)
    dis.apply(weights_init)

    optimG = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimD = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        gen.train()
        dis.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            bsz = imgs.size(0)

            real_labels = torch.ones(bsz, device=device)
            fake_labels = torch.zeros(bsz, device=device)

            # ------------------
            # Train Discriminator
            # ------------------
            optimD.zero_grad()
            outputs_real = dis(imgs)
            loss_real = criterion(outputs_real, real_labels)

            noise = torch.randn(bsz, args.latent_dim, device=device)
            fake_imgs = gen(noise)
            outputs_fake = dis(fake_imgs.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            optimD.step()

            # ------------------
            # Train Generator
            # ------------------
            optimG.zero_grad()
            outputs_fake_for_g = dis(fake_imgs)
            g_loss = criterion(outputs_fake_for_g, real_labels)
            g_loss.backward()
            optimG.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        avg_d = running_d_loss / len(dataloader)
        avg_g = running_g_loss / len(dataloader)
        print(f"Epoch [{epoch}/{args.epochs}]  D_loss: {avg_d:.4f}  G_loss: {avg_g:.4f}")

        with torch.no_grad():
            gen.eval()
            fake = gen(fixed_noise).detach().cpu()
            # denormalize from [-1,1] to [0,1]
            fake = (fake + 1) / 2.0
            utils.save_image(fake, samples_dir / f"epoch_{epoch:03d}.png", nrow=8)

        # save checkpoints
        torch.save(gen.state_dict(), out_dir / 'generator.pth')
        torch.save(dis.state_dict(), out_dir / 'discriminator.pth')
        # append epoch losses to log
        try:
            with open(out_dir / 'train_log.txt', 'a') as f:
                f.write(f"{epoch},{avg_d:.6f},{avg_g:.6f}\n")
        except Exception:
            pass

    print('Training complete. Outputs saved to', out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Dataset root or folder')
    parser.add_argument('--output_dir', type=str, default='output', help='Where to save models and samples')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU even if CUDA available')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
