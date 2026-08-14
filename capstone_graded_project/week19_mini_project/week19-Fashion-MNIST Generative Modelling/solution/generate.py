import argparse
from pathlib import Path

import torch
from torchvision import utils

from train_dcgan import Generator


def generate(checkpoint_path, out_dir, n_samples=64, latent_dim=100, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)

    gen = Generator(latent_dim=latent_dim).to(device)
    gen.load_state_dict(torch.load(checkpoint_path, map_location=device))
    gen.eval()

    z = torch.randn(n_samples, latent_dim, device=device)
    with torch.no_grad():
        samples = gen(z).cpu()
        samples = (samples + 1) / 2.0

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    utils.save_image(samples, out_dir / 'generated.png', nrow=8)
    print('Saved generated samples to', out_dir / 'generated.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to generator.pth')
    parser.add_argument('--out_dir', default='output/samples', help='Output directory for images')
    parser.add_argument('--n_samples', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=100)
    args = parser.parse_args()

    generate(args.checkpoint, args.out_dir, args.n_samples, args.latent_dim)
