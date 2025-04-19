from datasets import load_dataset, Audio
from transformers import WhisperProcessor
import datasets

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
def load_data(split="train", sampling_rate=16000):
    # Add trust_remote_code=True to allow custom code execution
    dataset = load_dataset("PolyAI/minds14", "en-US", trust_remote_code=True)
    
    # Split dataset and select relevant columns
    dataset = dataset["train"].train_test_split(seed=42, shuffle=True, test_size=0.2)
    dataset = dataset.select_columns(["audio", "transcription"])

    # Initialize processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", task="transcribe")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Preprocess the data
    def prepare(example):
        audio = example["audio"]
        result = processor(audio=audio["array"], sampling_rate=audio["sampling_rate"], text=example["transcription"])
        result["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return result

    dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names)
    
    return dataset["train"] if split == "train" else dataset["test"], processor
