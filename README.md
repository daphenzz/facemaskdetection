# Face Mask Detection

Simple face mask detection project.

Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training (entry point)

```bash
python main.py train --data data/images --epochs 10 --batch_size 32 --out models/best.h5
# Use data augmentation
python main.py train --data data/images --use_generator --epochs 20 --batch_size 32 --out models/best.h5
# Run with a fixed seed for reproducibility
python main.py train --data data/images --seed 123 --epochs 10
# Enable TensorBoard logging
python main.py train --data data/images --logdir runs/exp1
```

Prediction

Single image:

```bash
python main.py predict --model models/best.h5 --image path/to/image.jpg
```

Directory:

```bash
python main.py predict --model models/best.h5 --dir data/images/with_mask
```

Run full pipeline (train -> evaluate -> visualize -> grad-cam):

```bash
python main.py pipeline --data data/images --out models/best.h5 --epochs 10
```

Timestamped results and overwrite

```bash
# Create a new results/<TIMESTAMP>/ folder for this run
python main.py pipeline --data data/images --out models/best.h5 --epochs 10 --timestamp

# Create a new timestamped folder and overwrite if it already exists
python main.py pipeline --data data/images --out models/best.h5 --epochs 10 --timestamp --overwrite
```

Generate montages (mask error / correct groups):

```bash
python scripts/generate_montages.py --model models/best.h5 --data data/images
```

Tests

```bash
pytest -q
```

Outputs

Pipeline outputs are saved under `results/<TIMESTAMP>/` (confusion matrix, reports, montages, grad-cam images).

Files of interest

- `main.py` — CLI entrypoint
- `run_pipeline.py` — orchestration script
- `models/` — model code, training and evaluation helpers
- `data/` — data loading and preprocessing
- `scripts/generate_montages.py` — make grouped montages
