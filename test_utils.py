import torch
from jiwer import wer

def evaluate_model(model, dataloader, processor, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    preds, refs = [], []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["labels"].size(0)
            total_samples += batch["labels"].size(0)

            predicted_ids = model.generate(batch["input_features"])
            labels = batch["labels"].clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            label_str = processor.batch_decode(labels, skip_special_tokens=True)
            preds.extend(pred_str)
            refs.extend(label_str)

    avg_loss = total_loss / total_samples
    wer_score = wer(refs, preds)
    return avg_loss, wer_score
