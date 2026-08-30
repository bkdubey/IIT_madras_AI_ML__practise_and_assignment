import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

class CGGenerator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, emb_dim=10, out_channels=1):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            View((-1, 128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        e = self.embedding(labels)
        x = torch.cat([z, e], dim=1)
        return self.net(x)

class CGDiscriminator(nn.Module):
    def __init__(self, n_classes=10, in_channels=1, emb_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, emb_dim)
        # project embedding to a single-channel spatial map (28x28)
        self.label_proj = nn.Linear(emb_dim, 28 * 28)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, 128 * 7 * 7)),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x, labels):
        # create label channel by projecting embedding to 28x28 map
        e = self.embedding(labels)
        e_map = self.label_proj(e).view(-1, 1, 28, 28)
        x = torch.cat([x, e_map], dim=1)
        return self.conv(x).view(-1)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self,x):
        return x.view(*self.shape)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    print('Using device', device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    dataset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    gen = CGGenerator(latent_dim=args.latent_dim).to(device)
    dis = CGDiscriminator().to(device)
    gen.apply(weights_init)
    dis.apply(weights_init)
    optG = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5,0.999))
    optD = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5,0.999))
    criterion = nn.BCEWithLogitsLoss()
    out_dir = Path(args.output_dir) / 'cgan'
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    fixed_labels = torch.tensor([i%10 for i in range(64)], device=device)
    for epoch in range(1,args.epochs+1):
        running_d=0.0; running_g=0.0
        for imgs, labels in dl:
            imgs = imgs.to(device)
            labels = labels.to(device)
            bsz = imgs.size(0)
            real = torch.ones(bsz, device=device)
            fake = torch.zeros(bsz, device=device)
            optD.zero_grad()
            out_real = dis(imgs, labels)
            loss_real = criterion(out_real, real)
            z = torch.randn(bsz, args.latent_dim, device=device)
            gen_imgs = gen(z, labels)
            out_fake = dis(gen_imgs.detach(), labels)
            loss_fake = criterion(out_fake, fake)
            d_loss = (loss_real+loss_fake)*0.5
            d_loss.backward(); optD.step()
            optG.zero_grad()
            out_fake_for_g = dis(gen_imgs, labels)
            g_loss = criterion(out_fake_for_g, real)
            g_loss.backward(); optG.step()
            running_d += d_loss.item(); running_g += g_loss.item()
        avg_d = running_d/len(dl); avg_g = running_g/len(dl)
        print(f"CGAN Epoch {epoch} D:{avg_d:.4f} G:{avg_g:.4f}")
        with torch.no_grad():
            gen.eval(); fake = gen(fixed_noise, fixed_labels).cpu()
            fake = (fake+1)/2.0
            utils.save_image(fake, out_dir / f'epoch_{epoch:03d}.png', nrow=8)
            torch.save(gen.state_dict(), out_dir / 'generator.pth')
            torch.save(dis.state_dict(), out_dir / 'discriminator.pth')
            with open(out_dir / 'train_log.txt','a') as f:
                f.write(f"{epoch},{avg_d},{avg_g}\n")
    print('CGAN done, outputs in', out_dir)

# reuse weights_init from dcgan
import importlib.util
from pathlib import Path as P

try:
    from train_dcgan import weights_init
except Exception:
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            except Exception:
                pass
        elif classname.find('BatchNorm') != -1:
            try:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            except Exception:
                pass

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data_dir', default='data')
    p.add_argument('--output_dir', default='output')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=100)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--force_cpu', action='store_true')
    args=p.parse_args()
    train(args)
