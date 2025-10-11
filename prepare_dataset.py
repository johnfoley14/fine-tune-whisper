import torchaudio, json
from pathlib import Path
from torch.utils.data import Dataset

class AudioTextDataset(Dataset):
    def __init__(self, json_path, processor, sampling_rate=16000):
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