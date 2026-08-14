from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pathlib import Path

readme = Path(__file__).resolve().parents[0] / 'README.md'
out_pdf = Path(__file__).resolve().parents[0] / 'submission.pdf'
img_path = Path(__file__).resolve().parents[0] / 'output' / 'samples' / 'epoch_010.png'

c = canvas.Canvas(str(out_pdf), pagesize=letter)
width, height = letter

# Title
c.setFont('Helvetica-Bold', 18)
c.drawString(72, height - 72, 'Week 19: Fashion-MNIST Generative Modelling - Submission')

# Read README
y = height - 100
c.setFont('Helvetica', 10)
if readme.exists():
    text = readme.read_text().splitlines()
    for line in text:
        if y < 120:
            c.showPage()
            y = height - 72
            c.setFont('Helvetica', 10)
        c.drawString(72, y, line[:110])
        y -= 12
else:
    c.drawString(72, y, 'README not found in solution folder')
    y -= 12

# Add sample image
if img_path.exists():
    try:
        img = ImageReader(str(img_path))
        iw, ih = img.getSize()
        aspect = ih / iw
        display_w = 400
        display_h = display_w * aspect
        if y - display_h < 72:
            c.showPage()
            y = height - 72
        c.drawImage(img, 72, y - display_h, width=display_w, height=display_h)
    except Exception as e:
        c.drawString(72, y, f'Failed to embed image: {e}')
else:
    c.drawString(72, y, 'Sample image not found: ' + str(img_path))

c.save()
print('Created', out_pdf)
