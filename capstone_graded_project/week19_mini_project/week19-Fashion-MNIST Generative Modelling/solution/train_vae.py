import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1,32,4,2,1),
            nn.ReLU(True),
            nn.Conv2d(32,64,4,2,1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64*7*7, latent_dim)
        self.fc_logvar = nn.Linear(64*7*7, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 64*7*7)
        self.dec = nn.Sequential(
            nn.Unflatten(1,(64,7,7)),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,1,4,2,1),
            nn.Sigmoid(),
        )
    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self,x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparam(mu, logvar)
        out = self.fc_dec(z)
        out = self.dec(out)
        return out, mu, logvar

def loss_fn(recon, x, mu, logvar):
    bce = nn.functional.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0), bce / x.size(0), kld / x.size(0)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print('Using device', device)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = VAE(latent_dim=args.latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    out_dir = Path(args.output_dir) / 'vae'
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        running=0.0
        for imgs, _ in dl:
            imgs = imgs.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(imgs)
            loss, bce, kld = loss_fn(recon, imgs, mu, logvar)
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running/len(dl)
        print(f"VAE Epoch {epoch} loss {avg:.4f}")
        with torch.no_grad():
            sample = next(iter(dl))[0][:64].to(device)
            recon,_,_ = model(sample)
            utils.save_image(utils.make_grid(recon.cpu(), nrow=8), out_dir / f'epoch_{epoch:03d}_recon.png')
            torch.save(model.state_dict(), out_dir / 'vae.pth')
            with open(out_dir / 'train_log.txt','a') as f:
                f.write(f"{epoch},{avg}\n")
    print('VAE training done, outputs in', out_dir)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output_dir', default='output')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--force_cpu', action='store_true')
    args=p.parse_args()
    train(args)
