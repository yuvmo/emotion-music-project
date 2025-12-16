import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import librosa
import numpy as np
import soundfile as sf

from .validation import validate_audio, normalize_audio, ValidationResult
from .emotion import AudioEmotionClassifier, AudioEmotionResult, get_emotion_classifier


@dataclass
class AudioProcessingResult:
    status: str
    reason: Optional[str] = None
    transcript: Optional[str] = None
    emotion: Optional[str] = None
    emotion_confidence: Optional[float] = None
    all_emotions: Optional[Dict[str, float]] = None
    validation_details: Optional[Dict[str, Any]] = None
    raw_text: Optional[str] = None
    duration: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "transcript": self.transcript,
            "emotion": self.emotion,
            "emotion_confidence": self.emotion_confidence,
            "all_emotions": self.all_emotions,
            "validation_details": self.validation_details,
            "duration": self.duration,
        }


class AudioProcessor:
    def __init__(self, whisper_model_size: str = "small", target_sr: int = 16000):
        self.target_sr = target_sr
        self.whisper_model_size = whisper_model_size
        self._whisper_model = None
        self._emotion_classifier = None
    
    def _load_whisper(self):
        if self._whisper_model is None:
            import whisper
            self._whisper_model = whisper.load_model(self.whisper_model_size)
        return self._whisper_model
    
    def _get_emotion_classifier(self) -> AudioEmotionClassifier:
        if self._emotion_classifier is None:
            self._emotion_classifier = get_emotion_classifier()
        return self._emotion_classifier
    
    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        return audio, sr
    
    def load_audio_from_bytes(self, audio_bytes: bytes, format: str = "ogg") -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            audio, sr = self.load_audio(temp_path)
            return audio, sr
        finally:
            os.unlink(temp_path)
    
    def transcribe(self, audio: np.ndarray, sr: int, language: str = "ru") -> str:
        whisper_model = self._load_whisper()
        audio_norm = normalize_audio(audio)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_norm, sr)
            temp_path = f.name
        
        try:
            result = whisper_model.transcribe(
                temp_path,
                language=language,
                task="transcribe",
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.3
            )
            return result["text"].strip()
        finally:
            os.unlink(temp_path)
    
    def process(
        self, 
        audio_path: str = None,
        audio_bytes: bytes = None,
        audio_format: str = "ogg",
        language: str = "ru"
    ) -> AudioProcessingResult:
        try:
            if audio_path:
                audio, sr = self.load_audio(audio_path)
            elif audio_bytes:
                audio, sr = self.load_audio_from_bytes(audio_bytes, audio_format)
            else:
                return AudioProcessingResult(status="error", reason="no_audio_provided")
            
            duration = len(audio) / sr if sr > 0 else 0.0
            
            validation = validate_audio(audio, sr)
            if not validation.is_valid:
                return AudioProcessingResult(
                    status="invalid_audio",
                    reason=validation.reason,
                    validation_details=validation.details,
                    duration=duration
                )
            
            emotion_classifier = self._get_emotion_classifier()
            emotion_result = emotion_classifier.classify(audio, sr)
            
            transcript = self.transcribe(audio, sr, language)
            
            if len(transcript.split()) < 2:
                return AudioProcessingResult(
                    status="invalid_audio",
                    reason="transcript_too_short",
                    emotion=emotion_result.emotion,
                    emotion_confidence=emotion_result.confidence,
                    all_emotions=emotion_result.all_emotions,
                    transcript=transcript,
                    validation_details=validation.details,
                    duration=duration
                )
            
            return AudioProcessingResult(
                status="ok",
                transcript=transcript,
                raw_text=transcript.lower(),
                emotion=emotion_result.emotion,
                emotion_confidence=emotion_result.confidence,
                all_emotions=emotion_result.all_emotions,
                validation_details=validation.details,
                duration=duration
            )
            
        except Exception as e:
            return AudioProcessingResult(status="error", reason=str(e))


_processor_instance: Optional[AudioProcessor] = None


def get_audio_processor(whisper_model_size: str = "small") -> AudioProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = AudioProcessor(whisper_model_size=whisper_model_size)
    return _processor_instance
