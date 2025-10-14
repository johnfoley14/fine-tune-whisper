from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import wandb, torch, argparse
from jiwer import wer
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from prepare_dataset import AudioTextDataset

# parse cmd line args
parser = argparse.ArgumentParser(description="Fine-tune Whisper model with LoRA or full finetuning")
parser.add_argument("--mode", choices=["lora", "full"], default="lora", help="Choose 'lora' for LoRA fine-tuning or 'full' for full model fine-tuning")
args = parser.parse_args()

# Load the processor for feature extraction and tokenization --- feature extractor maps raw audio to mel spectrogram
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# Initialize the data collator to pad variable-length audio/text inputs within a batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Initialize Weights & Biases for experiment tracking
wandb.init(
    project="whisper",  # Name of the project on wandb
)

# Set learning rates and warmup steps based on fine-tuning mode
learning_rate = 4e-6 if args.mode == "full" else 4e-5
warmup_steps = 100 if args.mode == "full" else 50

# Define training hyperparameters and settings
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=25,
    save_strategy="steps",
    save_steps=100,
    num_train_epochs=8,   # reduce epochs for small dataset
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    save_total_limit=2,
    report_to=["wandb"],  # Log metrics to Weights & Biases
    fp16=False,
    bf16=True,
    predict_with_generate=True,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    generation_max_length=128,
)

device = "mps" if torch.backends.mps.is_available() else "cpu" # MPS (metal performance shaders) for Mac GPUs, else "cpu" 
# Load the actual model weights and neural network
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

#choose fine-tuning mode
if args.mode == "lora":
    print("Using LoRA fine-tuning")

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
    model.config.use_cache = False  # Disable caching during training for LoRA

    # Wrap the base model with LoRA using the above config
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
else:
    print("Using full model fine-tuning")
    # unfreeze all model parameters
    for param in model.parameters():
        param.requires_grad = True

    # count trainable parameters manually
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

DEFAULT_GEN_KWARGS = {
    "max_length": 128,
    "num_beams": 3,
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.5,
    "length_penalty": 1.0,
    "do_sample": False,
}

forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

model.config.forced_decoder_ids = forced_decoder_ids
model.config.decoder_start_token_id = decoder_start_token_id

def compute_metrics(pred):
    labels = pred.label_ids
    labels[labels == -100] = processor.tokenizer.pad_token_id

    preds = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    refs = processor.batch_decode(labels, skip_special_tokens=True)

    return {"wer": wer(refs, preds)}

# --- Datasets ---
train_dataset = AudioTextDataset(json_path="processed_dataset/train.json", processor=processor)
val_dataset   = AudioTextDataset(json_path="processed_dataset/validation.json", processor=processor)
test_dataset  = AudioTextDataset(json_path="processed_dataset/test.json", processor=processor)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Only patch generation in full fine-tuning mode
if args.mode == "full":
    original_generate = model.generate

    def generate_with_forced_ids(*args, **kwargs):
        kwargs.update(DEFAULT_GEN_KWARGS)
        kwargs.setdefault("forced_decoder_ids", forced_decoder_ids)
        kwargs.setdefault("decoder_start_token_id", decoder_start_token_id)
        return original_generate(*args, **kwargs)

    model.generate = generate_with_forced_ids


# --- Evaluate before training ---
pre_eval = trainer.evaluate(eval_dataset=test_dataset)
print(f"Before fine-tuning → Loss: {pre_eval['eval_loss']:.4f}, WER: {pre_eval['eval_wer']:.3f}")

# --- Training ---
trainer.train()

# === Save the model after training ===
save_dir = f"models/fine_tuned_whisper_{args.mode}"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
print(f"Model and processor saved to {save_dir}")

# --- Evaluate after training ---
post_eval = trainer.evaluate(eval_dataset=test_dataset)
print(f"After fine-tuning → Loss: {post_eval['eval_loss']:.4f}, WER: {post_eval['eval_wer']:.3f}")
