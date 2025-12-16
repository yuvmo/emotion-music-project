import librosa
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str
    details: dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


def is_silent(audio: np.ndarray, threshold: float = 0.003, min_ratio: float = 0.05) -> bool:
    rms = librosa.feature.rms(y=audio)[0]
    speech_ratio = (rms > threshold).mean()
    return speech_ratio < min_ratio


def is_too_short(audio: np.ndarray, sr: int, min_duration: float = 1.0) -> bool:
    return len(audio) / sr < min_duration


def is_noisy(audio: np.ndarray, threshold: float = 0.4) -> bool:
    zcr = librosa.feature.zero_crossing_rate(audio)
    return zcr.mean() > threshold


def validate_audio(audio: np.ndarray, sr: int) -> ValidationResult:
    duration = len(audio) / sr
    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(audio)
    
    details = {
        "duration_sec": round(duration, 2),
        "rms_mean": round(rms.mean(), 4),
        "rms_max": round(rms.max(), 4),
        "zcr_mean": round(zcr.mean(), 4),
    }
    
    if is_silent(audio):
        return ValidationResult(is_valid=False, reason="silence", details=details)
    
    if is_too_short(audio, sr):
        return ValidationResult(is_valid=False, reason="too_short", details=details)
    
    if is_noisy(audio):
        return ValidationResult(is_valid=False, reason="too_noisy", details=details)
    
    return ValidationResult(is_valid=True, reason="ok", details=details)


def normalize_audio(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    return audio * (target_rms / rms)
