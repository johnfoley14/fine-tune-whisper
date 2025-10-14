import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from prepare_dataset import AudioTextDataset

# === Device ===
device = "mps" if torch.backends.mps.is_available() else "cpu"

# === Load processor ===
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# === Data collator ===
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# === Load test dataset ===
test_dataset = AudioTextDataset(json_path="processed_dataset/test.json", processor=processor)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=8,
    collate_fn=data_collator
)

# === Generation arguments ===
gen_kwargs = {
    "max_length": 128,
    "num_beams": 3,
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.5,
    "length_penalty": 1.0,
    "do_sample": False,
}

gen_kwargs2 = {
    "max_length": 128,
    "num_beams": 1,
    "do_sample": False,
}

# === Evaluation function ===
def evaluate(model, dataloader, processor, device, gen_kwargs):
    model.eval()
    total_loss = 0
    total_samples = 0
    preds, refs = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id

        with torch.no_grad():
            # compute loss
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["labels"].size(0)
            total_samples += batch["labels"].size(0)

            # generate predictions
            forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
            predicted_ids = model.generate(batch["input_features"], forced_decoder_ids=forced_decoder_ids, **gen_kwargs)
            pred_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(labels, skip_special_tokens=True)

            preds.extend(pred_str)
            refs.extend(label_str)

    avg_loss = total_loss / total_samples
    wer_score = wer(refs, preds)
    return avg_loss, wer_score

# === Load models ===
print("Loading base Whisper tiny model...")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)

print("Loading fine-tuned model...")
ft_model = WhisperForConditionalGeneration.from_pretrained("models/fine_tuned_whisper_full").to(device)

# === Evaluate base model ===
base_loss, base_wer = evaluate(base_model, test_loader, processor, device, gen_kwargs)
print(f"Base Whisper Tiny → Loss: {base_loss:.4f}, WER: {base_wer:.3f}")

# === Evaluate fine-tuned model ===
ft_loss, ft_wer = evaluate(ft_model, test_loader, processor, device, gen_kwargs)
print(f"Fine-tuned Whisper → Loss: {ft_loss:.4f}, WER: {ft_wer:.3f}")

# === Evaluate base model with different gen_kwargs ===
base_loss, base_wer = evaluate(base_model, test_loader, processor, device, gen_kwargs2)
print(f"Base Whisper Tiny with different gen_kwargs → Loss: {base_loss:.4f}, WER: {base_wer:.3f}")

# === Evaluate fine-tuned model with different gen_kwargs ===
ft_loss, ft_wer = evaluate(ft_model, test_loader, processor, device, gen_kwargs2)
print(f"Fine-tuned Whisper with different gen_kwargs → Loss: {ft_loss:.4f}, WER: {ft_wer:.3f}")
