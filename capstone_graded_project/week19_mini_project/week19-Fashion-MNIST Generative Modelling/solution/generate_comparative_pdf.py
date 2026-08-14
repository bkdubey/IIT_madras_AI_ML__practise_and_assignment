from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pathlib import Path
import glob

out_pdf = Path(__file__).resolve().parents[0] / 'comparative_report.pdf'
out_dir = Path(__file__).resolve().parents[0] / 'output'

models = {
    'AE': out_dir / 'ae',
    'VAE': out_dir / 'vae',
    'DCGAN': out_dir,  # dcgan samples in output/samples
    'cGAN': out_dir / 'cgan',
    'StyleMix': out_dir / 'stylemix',
}

c = canvas.Canvas(str(out_pdf), pagesize=landscape(letter))
w, h = landscape(letter)

c.setFont('Helvetica-Bold', 16)
c.drawString(40, h - 40, 'Week 19 Comparative Report — Generative Models on Fashion-MNIST')

y = h - 80
c.setFont('Helvetica', 10)

rows = []
for name, path in models.items():
    info = {'name': name, 'metrics': 'N/A', 'image': None}
    # find train_log
    log = path / 'train_log.txt'
    if name == 'DCGAN':
        log = out_dir / 'train_log.txt'
    if log.exists():
        try:
            last = log.read_text().strip().splitlines()[-1]
            info['metrics'] = last
        except Exception:
            info['metrics'] = 'error_reading_log'
    # find latest sample image
    img = None
    if name == 'DCGAN':
        samples = sorted(glob.glob(str(out_dir / 'samples' / 'epoch_*.png')))
        img = samples[-1] if samples else None
    else:
        imgs = sorted(glob.glob(str(path / 'epoch_*.png')) + glob.glob(str(path / '*recon.png')))
        img = imgs[-1] if imgs else None
    if img:
        info['image'] = img
    rows.append(info)

# Draw table of metrics
c.setFont('Helvetica-Bold', 12)
c.drawString(40, y, 'Model')
c.drawString(160, y, 'Final log (last epoch)')

c.setFont('Helvetica', 10)
y -= 18
for info in rows:
    c.drawString(40, y, info['name'])
    c.drawString(160, y, info['metrics'][:80])
    y -= 14
    if y < 120:
        c.showPage(); y = h - 80

# New page for images
c.showPage()
c.setFont('Helvetica-Bold', 14)
c.drawString(40, h - 40, 'Sample Outputs')
img_y = h - 80
img_x = 40
max_h = 200
for info in rows:
    if info['image']:
        try:
            img = ImageReader(info['image'])
            iw, ih = img.getSize()
            scale = min(300 / iw, max_h / ih)
            display_w = iw * scale
            display_h = ih * scale
            if img_x + display_w > w - 40:
                img_x = 40
                img_y -= (max_h + 40)
            if img_y - display_h < 40:
                c.showPage(); img_y = h - 80; img_x = 40
            c.drawImage(img, img_x, img_y - display_h, width=display_w, height=display_h)
            c.setFont('Helvetica', 10)
            c.drawString(img_x, img_y - display_h - 12, info['name'])
            img_x += display_w + 20
        except Exception as e:
            c.drawString(img_x, img_y, f"Failed to embed {info['name']}: {e}")
            img_x += 200
    else:
        c.drawString(img_x, img_y, f"No image for {info['name']}")
        img_x += 200

c.save()
print('Created', out_pdf)
