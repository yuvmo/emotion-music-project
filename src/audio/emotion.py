from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AudioEmotionResult:
    emotion: str
    confidence: float
    all_emotions: dict = None
    
    def __post_init__(self):
        if self.all_emotions is None:
            self.all_emotions = {}


EMOTION_MAPPING = {
    "hap": "happy",
    "sad": "sad", 
    "ang": "angry",
    "neu": "neutral",
    "happy": "happy",
    "sadness": "sad",
    "angry": "angry",
    "neutral": "neutral",
    "fear": "anxious",
    "surprise": "energetic",
    "disgust": "angry",
}

EMOTION_MUSIC_PROFILES = {
    "happy": {
        "valence": (0.6, 0.9),
        "energy": (0.5, 0.8),
        "danceability": (0.5, 0.8),
        "tempo": (100, 140),
        "description": "веселая, позитивная"
    },
    "sad": {
        "valence": (0.1, 0.4),
        "energy": (0.2, 0.5),
        "danceability": (0.2, 0.5),
        "tempo": (60, 100),
        "description": "грустная, меланхоличная"
    },
    "angry": {
        "valence": (0.2, 0.5),
        "energy": (0.7, 1.0),
        "danceability": (0.4, 0.7),
        "tempo": (120, 180),
        "description": "агрессивная, мощная"
    },
    "neutral": {
        "valence": (0.4, 0.6),
        "energy": (0.4, 0.6),
        "danceability": (0.4, 0.6),
        "tempo": (90, 120),
        "description": "спокойная, нейтральная"
    },
    "energetic": {
        "valence": (0.5, 0.8),
        "energy": (0.7, 1.0),
        "danceability": (0.6, 0.9),
        "tempo": (120, 160),
        "description": "энергичная, драйвовая"
    },
    "calm": {
        "valence": (0.4, 0.7),
        "energy": (0.1, 0.4),
        "danceability": (0.2, 0.5),
        "tempo": (60, 100),
        "description": "спокойная, расслабляющая"
    },
    "anxious": {
        "valence": (0.2, 0.4),
        "energy": (0.5, 0.8),
        "danceability": (0.3, 0.5),
        "tempo": (100, 140),
        "description": "тревожная, напряженная"
    },
}


class AudioEmotionClassifier:
    def __init__(self):
        self._model = None
    
    def _load_model(self):
        if self._model is None:
            from transformers import pipeline
            self._model = pipeline(
                "audio-classification",
                model="superb/hubert-large-superb-er"
            )
        return self._model
    
    def classify(self, audio: np.ndarray, sampling_rate: int = 16000) -> AudioEmotionResult:
        model = self._load_model()
        results = model({"array": audio, "sampling_rate": sampling_rate})
        
        all_emotions = {}
        for r in results:
            raw_label = r["label"]
            mapped_label = EMOTION_MAPPING.get(raw_label, "neutral")
            score = r["score"]
            
            if mapped_label in all_emotions:
                all_emotions[mapped_label] = max(all_emotions[mapped_label], score)
            else:
                all_emotions[mapped_label] = score
        
        top_result = results[0]
        raw_emotion = top_result["label"]
        mapped_emotion = EMOTION_MAPPING.get(raw_emotion, "neutral")
        
        return AudioEmotionResult(
            emotion=mapped_emotion,
            confidence=round(top_result["score"], 3),
            all_emotions=all_emotions
        )
    
    @staticmethod
    def get_music_profile(emotion: str) -> dict:
        return EMOTION_MUSIC_PROFILES.get(emotion, EMOTION_MUSIC_PROFILES["neutral"])


_classifier_instance: Optional[AudioEmotionClassifier] = None


def get_emotion_classifier() -> AudioEmotionClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = AudioEmotionClassifier()
    return _classifier_instance
