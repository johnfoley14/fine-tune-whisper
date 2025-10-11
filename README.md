# fine-tune-whisper
Fine tune whisper model on Trump audio dataset. Goal is to improve ASR capabilities of Whisper Tiny model on Trump speech.

# Results before training:
Test loss before fine-tuning: 3.4806

# Results after training:
Test loss after fine-tuning: 0.4680

# Downloading Dataset
Python 3.12.11 used
Setup venv and install requirements.txt

```bash
# Create and activate a virtual environment (macOS/zsh)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

Run preprocessing to download from Kaggle and prepare the dataset for Whisper:

```bash
python3 preprocess.py
```

# Fine Tune Model
```bash
python3 fine_tune.py
```
