import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

class SMGenerator(nn.Module):
    def __init__(self, latent_dim=100, out_channels=1):
        super().__init__()
        # simple generator with intermediate features where we'll mix styles
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128*7*7), nn.ReLU(True))
        self.conv1 = nn.Sequential(nn.Unflatten(1,(128,7,7)), nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True))
        self.conv3 = nn.Conv2d(32,out_channels,3,1,1)
        self.tanh = nn.Tanh()
    def forward(self,z):
        h = self.fc(z)
        h = self.conv1(h)
        h = self.conv2(h)
        out = self.conv3(h)
        return self.tanh(out)

# style-mixing: generate two latents and swap a prefix to mix

def style_mix(z1, z2, split=50):
    # simple mixing by replacing first split dims
    z = z1.clone()
    z[:,:split] = z2[:,:split]
    return z


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print('Using device', device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    gen = SMGenerator(latent_dim=args.latent_dim).to(device)
    opt = optim.Adam(gen.parameters(), lr=args.lr)
    out_dir = Path(args.output_dir) / 'stylemix'
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1,args.epochs+1):
        gen.train()
        for imgs,_ in dl:
            imgs = imgs.to(device)
            # dummy reconstruction/reg loss to train generator to produce images
            z = torch.randn(imgs.size(0), args.latent_dim, device=device)
            out = gen(z)
            loss = ((out - torch.randn_like(out)*0.1).pow(2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"StyleMix Epoch {epoch} dummy loss {loss.item():.4f}")
        # create style-mix samples
        with torch.no_grad():
            z1 = torch.randn(64, args.latent_dim, device=device)
            z2 = torch.randn(64, args.latent_dim, device=device)
            mixed = style_mix(z1,z2, split=args.split)
            samples = gen(mixed).cpu()
            samples = (samples+1)/2.0
            utils.save_image(samples, out_dir / f'epoch_{epoch:03d}.png', nrow=8)
            torch.save(gen.state_dict(), out_dir / 'stylemix.pth')
            with open(out_dir / 'train_log.txt','a') as f:
                f.write(f"{epoch},{loss.item()}\n")
    print('Style-mix done, outputs in', out_dir)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output_dir', default='output')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=100)
    p.add_argument('--split', type=int, default=50)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--force_cpu', action='store_true')
    args=p.parse_args()
    train(args)
