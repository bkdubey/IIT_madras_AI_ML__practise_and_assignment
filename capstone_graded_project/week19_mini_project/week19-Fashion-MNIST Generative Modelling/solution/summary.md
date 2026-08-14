# Training Summary

- Model: DCGAN-style generator + discriminator
- Dataset: Fashion-MNIST (28x28 grayscale)
- Device: MPS (Apple GPU) when available
- Hyperparameters: epochs=50, batch_size=128, latent_dim=100, lr=2e-4

## Quantitative Results
- Final discriminator loss (epoch 50): 0.393384
- Final generator loss (epoch 50): 1.770139

## Artifacts
- Checkpoints: `output/generator.pth`, `output/discriminator.pth`
- Per-epoch sample grids: `output/samples/epoch_001.png` ... `epoch_050.png`
- Training log: `output/train_log.txt`

## Notes
- Losses show generator improving stability over training; visual quality should be reviewed in the sample grids.
- For production-quality results, train on a more powerful GPU for more epochs and consider GAN improvements (WGAN-GP, spectral norm, conditional GAN).
