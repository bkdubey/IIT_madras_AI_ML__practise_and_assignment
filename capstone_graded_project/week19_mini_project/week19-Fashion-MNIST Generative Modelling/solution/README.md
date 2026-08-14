# Fashion-MNIST Generative Modelling Solution

This folder contains a PyTorch implementation of a DCGAN-style model trained on Fashion-MNIST.

Files:
- `train_dcgan.py`: Train a DCGAN on Fashion-MNIST. Downloads data to the provided data directory if missing.
- `generate.py`: Generate sample images from a saved generator checkpoint.
- `requirements.txt`: Python dependencies.

Quick start:

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r capstone_graded_project/week19_mini_project/week19-Fashion-MNIST\ Generative\ Modelling/solution/requirements.txt
```

2. Train (downloads Fashion-MNIST into the specified folder if needed):

```bash
python capstone_graded_project/week19_mini_project/week19-Fashion-MNIST\ Generative\ Modelling/solution/train_dcgan.py \
  --data_dir capstone_graded_project/week19_mini_project/week19-Fashion-MNIST\ Generative\ Modelling/prob_definition_with_dataset/input_dataset \
  --output_dir capstone_graded_project/week19_mini_project/week19-Fashion-MNIST\ Generative\ Modelling/solution/output \
  --epochs 20
```

3. Generate samples:

```bash
python capstone_graded_project/week19_mini_project/week19-Fashion-MNIST\ Generative\ Modelling/solution/generate.py \
  --checkpoint solution/output/generator.pth \
  --out_dir solution/output/samples
```

Notes:
- The scripts are set to work on CPU or CUDA if available.
- If you prefer using a pre-downloaded dataset, place it under the `input_dataset` folder; the training script will use it.
