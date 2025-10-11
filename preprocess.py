
import json, random, re
from pathlib import Path
from pydub import AudioSegment
import kagglehub

# --- Download Kaggle dataset ---
path = kagglehub.dataset_download("etaifour/trump-speeches-audio-and-word-transcription")

# --- Settings ---
output_dir = Path("processed_dataset")
split_ratios = (0.8, 0.1, 0.1) # train, val, test
min_len, max_len = 5.0, 30.0  # max and min length of audio segements in seconds
random.seed(42)

output_dir.mkdir(parents=True, exist_ok=True)
meta = []

dataset_root = Path(path)

def segment_words(word_items, min_len=5.0, max_len=30.0):
    """
    Split list of word items [{'startTime', 'endTime', 'value'}] into 5-30s segments.
    Prefers sentence or comma breaks, falls back to receptive windows.
    """
    segments = []
    current = []
    seg_start = None

    for w in word_items:
        start, end, text = float(w["startTime"]), float(w["endTime"]), w["value"]
        if seg_start is None:
            seg_start = start
        current.append((start, end, text))

        duration = end - seg_start
        # ensure each segment is over 5 secs
        if duration >= min_len and re.search(r'[.?!]', text):
            # 
            if duration <= max_len:
                seg_text = ''.join(t for _, _, t in current).strip()
                segments.append((seg_start, end, seg_text))
                current, seg_start = [], None
            else:
                # too long — try to break at commas
                seg_text = ''.join(t for _, _, t in current).strip()
                parts = re.split(r'[,;]', seg_text)
                t0 = seg_start
                chunk_len = (end - seg_start) / len(parts)
                for p in parts:
                    if not p.strip():
                        continue
                    t1 = min(t0 + chunk_len, end)
                    if t1 - t0 >= min_len:
                        segments.append((t0, t1, p.strip()))
                    t0 = t1
                current, seg_start = [], None

    # handle leftover words
    if current:
        start, end = current[0][0], current[-1][1]
        seg_text = ''.join(t for _, _, t in current).strip()
        if end - start > max_len:
            # fallback receptive windows
            t0 = start
            while t0 < end:
                t1 = min(t0 + max_len, end)
                segments.append((t0, t1, seg_text))
                if t1 == end:
                    break
                t0 += max_len * 0.8  # 20% overlap
        elif end - start >= min_len:
            segments.append((start, end, seg_text))

    return segments


for file in ["Trump_WEF_2018", "Trumps_speech_at_75th_d_day_anniversary_in_normandy_full_remarks_UhOMVlQxapY", "state of the union 2018", "state-of-the-union-trump_2019-02-05-225820-8225-0-0-0.64kmono", "trump_speech_in_miami_about_venezuela_and_socialism_21819_BVrdTed4z2M"]:
    audio_file = dataset_root / f"{file}.mp3"
    json_file = audio_file.with_name(audio_file.name + ".json")

    if not json_file or not json_file.exists():
        print(f"⚠️ Skipping {audio_file.name}, no matching transcript.")
        continue
    audio = AudioSegment.from_file(audio_file)
    with open(json_file, "r", encoding="utf-8-sig") as f:
        transcript = json.load(f)

    segments = segment_words(transcript["words"], min_len=5.0, max_len=30.0)

    for i, (start, end, text) in enumerate(segments):
        clip_name = f"{audio_file.stem}_{i:04d}.wav"
        clip_path = output_dir / "audio" / clip_name
        clip_path.parent.mkdir(exist_ok=True)

        clip = audio[start * 1000 : end * 1000]
        clip.export(clip_path, format="wav", parameters=["-ar", "16000", "-ac", "1"])

        meta.append({
            "audio": f"audio/{clip_name}",  # relative path
            "text": text.strip(),
            "duration": round(end - start, 3)
        })

# --- Split into train/val/test ---
random.shuffle(meta)
n = len(meta)
n_train = int(split_ratios[0] * n)
n_val = int(split_ratios[1] * n)

splits = {
    "train": meta[:n_train],
    "validation": meta[n_train:n_train + n_val],
    "test": meta[n_train + n_val:]
}

for split, items in splits.items():
    with open(output_dir / f"{split}.json", "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f)
            f.write("\n")  # JSONL format

print(f"\n✅ Created {len(meta)} total segments.")
print(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
print(f"Processed dataset saved at: {output_dir.resolve()}")
