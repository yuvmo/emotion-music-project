import whisper

_model = None

def load_stt_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model

def transcribe(audio_path: str) -> str:
    model = load_stt_model()
    result = model.transcribe(audio_path, language="ru")
    return result["text"]