This directory vendors the public MolFM QM9 sampling bundle used by the local
`molfm` backend integration.

Source:
- repo: `https://github.com/GenSI-THUAIR/MolFM`
- commit: `2c83c0f065e12a79e9206035235c72e06cbb279c`
- upstream files copied from `sampling/`:
  - `args.pickle`
  - `generative_model_ema_0.npy`

Notes:
- The upstream repo also ships a Docker-based environment setup, but this repo's
  MolFM integration does not require running that container for post-training.
- The checkpoint is QM9-only. There is no public GEOM MolFM checkpoint vendored
  here.
