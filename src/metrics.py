import csv
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import threading


METRICS_FILE = Path(__file__).parent.parent / "data" / "metrics.csv"
_lock = threading.Lock()


@dataclass
class RequestMetrics:
    request_id: str = ""
    user_id: int = 0
    timestamp: str = ""
    
    audio_duration_sec: float = 0.0
    processing_time_sec: float = 0.0
    
    audio_valid: bool = False
    validation_error: str = ""
    
    transcript: str = ""
    transcript_length: int = 0
    stt_time_sec: float = 0.0
    
    emotion: str = ""
    emotion_confidence: float = 0.0
    emotion_time_sec: float = 0.0
    
    intents_genre: str = ""
    intents_language: str = ""
    intents_count: int = 0
    
    llm_success: bool = False
    llm_time_sec: float = 0.0
    target_valence: float = 0.0
    target_energy: float = 0.0
    target_danceability: float = 0.0
    target_tempo: float = 0.0
    
    tracks_found: int = 0
    tracks_from_dataset: int = 0
    tracks_from_spotify: int = 0
    
    success: bool = False
    error: str = ""


class MetricsCollector:
    def __init__(self):
        self._current: Optional[RequestMetrics] = None
        self._start_time: float = 0
        self._step_times: dict = {}
    
    def start_request(self, user_id: int) -> RequestMetrics:
        self._current = RequestMetrics(
            request_id=f"{user_id}_{int(time.time()*1000)}",
            user_id=user_id,
            timestamp=datetime.now().isoformat()
        )
        self._start_time = time.time()
        self._step_times = {}
        return self._current
    
    def start_step(self, step_name: str):
        self._step_times[step_name] = time.time()
    
    def end_step(self, step_name: str) -> float:
        if step_name in self._step_times:
            elapsed = time.time() - self._step_times[step_name]
            return elapsed
        return 0.0
    
    def finalize(self) -> RequestMetrics:
        if self._current:
            self._current.processing_time_sec = round(time.time() - self._start_time, 3)
        return self._current
    
    def save(self):
        if not self._current:
            return
        
        with _lock:
            file_exists = METRICS_FILE.exists()
            
            with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        "request_id", "user_id", "timestamp",
                        "audio_duration_sec", "processing_time_sec",
                        "audio_valid", "validation_error",
                        "transcript", "transcript_length", "stt_time_sec",
                        "emotion", "emotion_confidence", "emotion_time_sec",
                        "intents_genre", "intents_language", "intents_count",
                        "llm_success", "llm_time_sec",
                        "target_valence", "target_energy", "target_danceability", "target_tempo",
                        "tracks_found", "tracks_from_dataset", "tracks_from_spotify",
                        "success", "error"
                    ])
                
                m = self._current
                writer.writerow([
                    m.request_id, m.user_id, m.timestamp,
                    m.audio_duration_sec, m.processing_time_sec,
                    m.audio_valid, m.validation_error,
                    m.transcript[:200], m.transcript_length, m.stt_time_sec,
                    m.emotion, m.emotion_confidence, m.emotion_time_sec,
                    m.intents_genre, m.intents_language, m.intents_count,
                    m.llm_success, m.llm_time_sec,
                    m.target_valence, m.target_energy, m.target_danceability, m.target_tempo,
                    m.tracks_found, m.tracks_from_dataset, m.tracks_from_spotify,
                    m.success, m.error
                ])


_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector

