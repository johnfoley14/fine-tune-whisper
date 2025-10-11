import torchaudio, json
from pathlib import Path
from torch.utils.data import Dataset
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

class AudioTextDataset(Dataset):
    def __init__(self, json_path, processor=processor, sampling_rate=16000):
        """
        Initializes the dataset with the path to the JSON file and processor.

        Args:
            json_path (str): Path to the JSONL file containing audio-transcript pairs.
            processor: Processor object for tokenizing text and preparing audio features.
            sampling_rate (int): Sampling rate for audio data.
        """
        self.json_path = str(json_path)
        self.base_dir = Path(self.json_path).parent
        # The processed dataset is JSONL (one JSON object per line)
        self.data = []
        with open(self.json_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item by index and processes it.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing input features and labels for the model.
        """
        item = self.data[idx]
        # processed_dataset JSONL format: {"audio": "audio/<file>.wav", "text": "<transcript>", "duration": <seconds>}
        audio_rel = item["audio"]
        transcript = item["text"]
        audio_path = str((self.base_dir / audio_rel).resolve())

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)(waveform)
        waveform = waveform.squeeze(0).numpy()

        # Process inputs and labels
        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        labels = self.processor.tokenizer(transcript, return_tensors="pt").input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0),
            "transcript": transcript
        }




##################################################################################################################




import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for sequence-to-sequence speech tasks using Whisper.

    This collator dynamically pads both the input audio features and the target text tokens
    to the maximum length in a batch, making it compatible with variable-length input/output sequences.

    Attributes:
        processor (Any): A Hugging Face `WhisperProcessor` that includes both a feature extractor
                         for audio and a tokenizer for text.

    Methods:
        __call__(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            Pads and collates a batch of audio-text pairs for model input.
    """

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Pads input audio features and target text labels for a batch of samples.

        Args:
            features (List[Dict]): Each item in the list is a dictionary with:
                - 'input_features': Audio features (from spectrogram extraction)
                - 'labels': Tokenized text labels

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_features': Padded audio features
                - 'labels': Padded and masked labels (with padding tokens replaced by -100)
        """

        # Pad audio features
        input_features = [{"input_features": feat["input_features"]} for feat in features]
        batch = self.processor.feature_extractor.pad(
            input_features, 
            padding=True,
            return_tensors="pt"
        )

        # Pad text labels
        labels = [{"input_ids": feat["labels"]} for feat in features]
        labels_batch = self.processor.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt"
        )

        # Replace padding token IDs with -100 so they are ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Optionally remove BOS token if present at the beginning
        if (
            labels.size(1) > 1
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch




##################################################################################################################




from transformers import WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import prepare_model_for_kbit_training
import wandb

# Initialize the data collator to pad variable-length audio/text inputs
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Initialize Weights & Biases for experiment tracking
wandb.init(
    project="whisper",  # Name of the project on wandb
)

# Define training hyperparameters and settings
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",  # Directory to save model checkpoints
    per_device_train_batch_size=8,  # Batch size per GPU
    gradient_accumulation_steps=1,  # Accumulate gradients for effective batch size
    learning_rate=1e-3,  # Learning rate
    warmup_steps=0,  # Number of warmup steps for learning rate scheduler
    num_train_epochs=3,  # Total number of training epochs
    eval_strategy="steps",  # Evaluate every few steps
    logging_strategy="steps",  # Log every few steps
    logging_first_step=True,  # Log the very first training step
    logging_nan_inf_filter=False,  # Don’t filter NaN/inf in logs
    eval_steps=500,  # Run evaluation every 500 steps
    report_to=["wandb"],  # Log metrics to Weights & Biases
    fp16=False,  # Use mixed-precision (FP16) training
    bf16=True,  # Use mixed-precision (FP16) training --- IGNORE ---
    per_device_eval_batch_size=8,  # Batch size for evaluation
    generation_max_length=128,  # Max length for generation during eval
    logging_steps=1,  # Log every step
    remove_unused_columns=False,  # Needed for PEFT since forward signature is modified
    label_names=["labels"],  # Tells Trainer to pass labels explicitly
)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device: ", device)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

# Prepare the model for LoRA-compatible 8-bit training (freezing norms, casting types)
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
config = LoraConfig(
    r=32,  # Rank of LoRA decomposition
    lora_alpha=64,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention projections
    lora_dropout=0.05,  # Dropout applied to LoRA layers
    bias="none"  # Don't adapt bias terms
)

# Wrap the base model with LoRA using the above config
model = get_peft_model(model, config)
model.print_trainable_parameters()  # Print which parameters are trainable

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"eval_loss": trainer.evaluate()["eval_loss"]}

# Initialize Hugging Face Trainer for training and evaluation
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=AudioTextDataset(json_path="processed_dataset/train.json", processor=processor),
    eval_dataset=AudioTextDataset(json_path="processed_dataset/validation.json", processor=processor),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,  # Optional; may be unused
)

# Disable caching to avoid warnings during training (re-enable for inference)
model.config.use_cache = False

def test_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    for batch in dataloader:
        # Move everything to device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            # Do NOT pass labels=labels separately
            outputs = model(**batch)
            # Loss is already computed internally
            total_loss += outputs.loss.item() * batch["labels"].size(0)
            total_samples += batch["labels"].size(0)
    return total_loss / total_samples

# === Load test dataset ===
test_dataset = AudioTextDataset(json_path="processed_dataset/test.json", processor=processor)

# Use collator to handle padding
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    collate_fn=data_collator
)

# === Evaluate before training ===
pre_train_loss = test_model(model, test_loader, device)
print(f"Test loss before fine-tuning: {pre_train_loss:.4f}")

# === Training ===
trainer.train()

# === Save the model after training ===
save_dir = "fine_tuned_whisper"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print(f"Model and processor saved to {save_dir}")

# === Evaluate after training ===
post_train_loss = test_model(model, test_loader, device)
print(f"Test loss after fine-tuning: {post_train_loss:.4f}")
