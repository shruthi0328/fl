from transformers import WhisperForConditionalGeneration

def load_model():
   return  WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny", 
        use_auth_token="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"
    )
