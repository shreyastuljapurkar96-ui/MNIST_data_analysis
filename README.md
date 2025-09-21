# How to run

1) Put raw TXT files into data/raw/ (or edit absolute paths in config.yaml).
2) Create venv and install requirements:
   python -m venv .venv
   .venv/Scripts/activate  (Windows)   |   source .venv/bin/activate  (macOS/Linux)
   pip install -r requirements.txt
3) Build compact features + labels (subset mode on by default):
   python -m src.build_features --config config.yaml
4) Train, calibrate, evaluate, and plot:
   python -m src.train_eval
5) Turn off subset in config.yaml to process full dataset; rerun steps 3–4.
