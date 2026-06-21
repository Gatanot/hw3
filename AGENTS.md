# AGENTS.md

## Project

FixMatch semi-supervised image classification on CIFAR-10 using WideResNet-28-2 (PyTorch).
No tests, linting, typecheck, or CI in this repo.

## Entry points

- **Single run**: `python3 train.py --num_labels 40 [--use_amp --cudnn_benchmark]`
- **All-in-one**: `./run_all.sh` (runs 40 → 250 → 4000 sequentially, ~15h on RTX 4090)
- Always run from repo root — imports use `from src.xxx import yyy`

## Key commands (from `DEPLOY.md`)

```bash
# Environment
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Verify GPU
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Configuration

- `config/default.yaml` is auto-loaded; CLI flags override YAML values
- CLI args: `--num_labels` (40/250/4000), `--use_amp`, `--cudnn_benchmark`, `--use_interleave`

## Gotchas

- **Data must pre-exist**: `dataset.py` hardcodes `download=False`. Run torchvision once or download CIFAR-10 to `./data/` first. See `experiments/00_commands.md` line 8 for the one-time download command.
- **`drop_last` asymmetry**: `labeled_loader` uses `drop_last=False` (critical for num_labels=40 — only 1 batch), `unlabeled_loader` uses `drop_last=True`
- **`use_interleave`**: defaults `True` in `fixmatch_loss()` but `False` in argparse. Single GPU: use `False`. The `fixmatch_loss` function has a fallback when batch sizes don't align for interleave.
- **EMA eval pattern**: `ema.apply_shadow()` → evaluate → `ema.restore()` → `model.train()`. Always restore before resuming training.
- **No resume support**: training state is saved but there's no code to load and resume from checkpoints.
- **`infor.md`** contains SSH credentials — never read or expose it.

## Verification snippets

See `experiments/00_commands.md` for one-liner verification commands (data split, DataLoader shapes, WRN forward pass, FixMatch loss on a single batch, CPU 10-step smoke test).

## Training artifacts

- Checkpoints: `checkpoints/fixmatch_<N>/best_model.pth` (EMA weights) + `checkpoint_*.pth`
- Logs: `logs/fixmatch_<N>/` (TensorBoard events + stdout)
- Extract results: `grep "best_acc\|Final" logs/fixmatch_*/train.log`

## Report

LaTeX report at `report/main.tex`. Build with `pdflatex`/`latexmk`.
