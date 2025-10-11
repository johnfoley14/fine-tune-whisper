from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import wandb, torch
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from prepare_dataset import AudioTextDataset
from test_utils import evaluate_model

# Load the processor for feature extraction and tokenization --- feature extractor maps raw audio to mel spectrogram
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# Initialize the data collator to pad variable-length audio/text inputs within a batch
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

device = "mps" if torch.backends.mps.is_available() else "cpu" # MPS (metal performance shaders) for Mac GPUs, else "cpu"
# Load the actual model weights and neural network
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

# === Load test dataset ===
test_dataset = AudioTextDataset(json_path="processed_dataset/test.json", processor=processor)

# Load test dataset and use collator to handle padding
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    collate_fn=data_collator
)

# === Evaluate before training ===
pre_loss, pre_wer = evaluate_model(model, test_loader, processor, device)
print(f"Before fine-tuning → Loss: {pre_loss:.4f}, WER: {pre_wer:.3f}")

# === Training ===
trainer.train()

# === Save the model after training ===
save_dir = "fine_tuned_whisper"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print(f"Model and processor saved to {save_dir}")

# === Evaluate after training ===
post_loss, post_wer = evaluate_model(model, test_loader, processor, device)
print(f"After fine-tuning → Loss: {post_loss:.4f}, WER: {post_wer:.3f}")
