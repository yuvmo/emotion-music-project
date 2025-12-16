import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.audio.processor import AudioProcessor, get_audio_processor, AudioProcessingResult
from src.intent.extractor import extract_user_intent, UserIntent
from src.llm.gigachat import GigaChatService, get_gigachat_service
from src.recommender.music import MusicRecommender, get_music_recommender, Track
from src.spotify_client import SpotifyClient, get_spotify_client
from src.track_validator import TrackValidator, get_track_validator, ValidatedTrack
from src.metrics import get_collector

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    error_message: Optional[str] = None
    response_text: str = ""
    tracks: List[Track] = field(default_factory=list)
    transcript: Optional[str] = None
    audio_emotion: Optional[str] = None
    intent: Optional[UserIntent] = None
    mood_interpretation: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    filters: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "error_message": self.error_message,
            "response_text": self.response_text,
            "tracks": [t.to_dict() for t in self.tracks],
            "transcript": self.transcript,
            "audio_emotion": self.audio_emotion,
            "mood_interpretation": self.mood_interpretation,
        }


class MusicRecommendationPipeline:
    def __init__(
        self,
        audio_processor: Optional[AudioProcessor] = None,
        gigachat_service: Optional[GigaChatService] = None,
        music_recommender: Optional[MusicRecommender] = None,
        spotify_client: Optional[SpotifyClient] = None,
        track_validator: Optional[TrackValidator] = None,
        use_spotify_search: bool = True,
        validate_tracks: bool = True
    ):
        self.audio_processor = audio_processor or get_audio_processor()
        self.gigachat_service = gigachat_service or get_gigachat_service()
        self.music_recommender = music_recommender or get_music_recommender()
        self.spotify_client = spotify_client or get_spotify_client()
        self.track_validator = track_validator or get_track_validator()
        self.use_spotify_search = use_spotify_search
        self.validate_tracks_flag = validate_tracks
    
    def process_audio(
        self,
        audio_path: str = None,
        audio_bytes: bytes = None,
        audio_format: str = "ogg",
        top_k: int = 5,
        user_id: int = 0
    ) -> PipelineResult:
        metrics = get_collector()
        m = metrics.start_request(user_id)
        
        logger.info("Step 1: Audio processing...")
        metrics.start_step("audio")
        
        audio_result = self.audio_processor.process(
            audio_path=audio_path,
            audio_bytes=audio_bytes,
            audio_format=audio_format
        )
        
        m.audio_duration_sec = audio_result.duration or 0.0
        m.stt_time_sec = metrics.end_step("audio")
        
        if audio_result.status != "ok":
            m.audio_valid = False
            m.validation_error = audio_result.reason or ""
            m.success = False
            m.error = audio_result.reason or "invalid_audio"
            metrics.finalize()
            metrics.save()
            
            error_messages = {
                "invalid_audio": self._get_invalid_audio_message(audio_result.reason),
                "error": f"Error: {audio_result.reason}"
            }
            return PipelineResult(
                success=False,
                error_message=error_messages.get(audio_result.status, "Unknown error"),
                transcript=audio_result.transcript,
                audio_emotion=audio_result.emotion
            )
        
        m.audio_valid = True
        m.transcript = audio_result.transcript or ""
        m.transcript_length = len(m.transcript)
        m.emotion = audio_result.emotion or ""
        m.emotion_confidence = audio_result.emotion_confidence or 0.0
        
        logger.info(f"Transcript: {audio_result.transcript}")
        logger.info(f"Emotion: {audio_result.emotion} ({audio_result.emotion_confidence})")
        
        logger.info("Step 2: Intent extraction...")
        
        intent = extract_user_intent(
            text=audio_result.transcript,
            audio_emotion=audio_result.emotion,
            audio_emotion_confidence=audio_result.emotion_confidence
        )
        
        m.intents_genre = ",".join(intent.genres) if intent.genres else ""
        m.intents_language = intent.language or ""
        m.intents_count = len(intent.genres) + (1 if intent.language else 0) + (1 if intent.artist else 0)
        
        logger.info(f"Genres: {intent.genres}")
        logger.info(f"Language: {intent.language}")
        logger.info(f"Artist: {intent.artist}")
        
        needs_clarification = (
            len(audio_result.transcript or "") < 5 and
            not intent.genres and
            not intent.mood_keywords and
            (audio_result.emotion_confidence or 0) < 0.5
        )
        
        if needs_clarification:
            logger.info("Request is unclear, asking for clarification")
            clarification = self.gigachat_service.generate_clarification(intent)
            m.success = True
            metrics.finalize()
            metrics.save()
            return PipelineResult(
                success=True,
                response_text=clarification,
                tracks=[],
                transcript=audio_result.transcript,
                audio_emotion=audio_result.emotion,
                intent=intent,
            )
        
        logger.info("Step 3: GigaChat analysis...")
        metrics.start_step("llm")
        
        analysis = self.gigachat_service.analyze_music_request(intent)
        
        m.llm_time_sec = metrics.end_step("llm")
        m.llm_success = bool(analysis)
        
        features = analysis.get("features", {})
        filters = analysis.get("filters", {})
        mood_interpretation = analysis.get("mood_interpretation", "")
        
        m.target_valence = features.get("valence", 0.0)
        m.target_energy = features.get("energy", 0.0)
        m.target_danceability = features.get("danceability", 0.0)
        m.target_tempo = features.get("tempo", 0.0)
        
        logger.info(f"Interpretation: {mood_interpretation}")
        
        if filters.get("artist"):
            logger.info(f"Artist: {filters.get('artist')}")
        
        if intent.language and not filters.get("language"):
            filters["language"] = intent.language
        if intent.genres and not filters.get("genres"):
            filters["genres"] = intent.genres
        
        logger.info("Step 4: Track recommendation...")
        
        tracks = self.music_recommender.recommend(
            features=features,
            filters=filters,
            top_k=top_k
        )
        
        m.tracks_from_dataset = len(tracks)
        logger.info(f"Found {len(tracks)} tracks from local dataset")
        
        if self.use_spotify_search and self.spotify_client.is_available() and len(tracks) < top_k:
            spotify_tracks = self._search_spotify_tracks(intent, features, top_k - len(tracks))
            if spotify_tracks:
                tracks.extend(spotify_tracks)
                m.tracks_from_spotify = len(spotify_tracks)
                logger.info(f"Added {len(spotify_tracks)} tracks from Spotify search")
        
        if self.validate_tracks_flag and tracks:
            logger.info("Step 4b: Validating tracks...")
            track_dicts = [t.to_dict() for t in tracks]
            validated = self.track_validator.validate_and_enrich(
                track_dicts,
                max_spotify_calls=3,
                use_gigachat=False
            )
            
            tracks = []
            for v in validated:
                track = Track(
                    spotify_id=v.spotify_id,
                    name=v.name,
                    artist=v.artist,
                    year=v.year,
                    genres=v.genres,
                    language=v.language,
                )
                tracks.append(track)
            
            verified_count = sum(1 for v in validated if v.is_verified)
            logger.info(f"Validated {len(tracks)} tracks, {verified_count} verified via Spotify")
        
        m.tracks_found = len(tracks)
        
        logger.info("Step 5: Response generation...")
        
        response_text = self.gigachat_service.generate_response(
            tracks=[t.to_dict() for t in tracks],
            intent=intent,
            mood_interpretation=mood_interpretation
        )
        
        m.success = True
        metrics.finalize()
        metrics.save()
        
        return PipelineResult(
            success=True,
            response_text=response_text,
            tracks=tracks,
            transcript=audio_result.transcript,
            audio_emotion=audio_result.emotion,
            intent=intent,
            mood_interpretation=mood_interpretation,
            features=features,
            filters=filters
        )
    
    def _get_invalid_audio_message(self, reason: str) -> str:
        messages = {
            "silence": "В сообщении тишина. Попробуй записать еще раз.",
            "too_short": "Сообщение слишком короткое. Расскажи подробнее, какую музыку хочешь.",
            "too_noisy": "Слишком много шума, не могу разобрать. Попробуй в более тихом месте.",
            "transcript_too_short": "Не удалось распознать речь. Попробуй еще раз.",
            "no_audio_provided": "Отправь голосовое сообщение, и я подберу музыку."
        }
        return messages.get(reason, "Что-то пошло не так. Попробуй еще раз.")
    
    def _search_spotify_tracks(
        self,
        intent: UserIntent,
        features: Dict[str, float],
        limit: int
    ) -> List[Track]:
        mood_map = {
            "happy": "happy",
            "sad": "sad",
            "angry": "energetic",
            "neutral": "calm",
            "fear": "calm",
            "disgust": "sad",
            "surprise": "energetic"
        }
        
        mood = mood_map.get(intent.audio_emotion, "")
        genre = intent.genres[0] if intent.genres else None
        language = intent.language
        
        if mood:
            spotify_results = self.spotify_client.search_by_mood(
                mood=mood,
                genre=genre,
                language=language,
                limit=limit
            )
        else:
            query_parts = []
            if genre:
                query_parts.append(genre)
            if intent.keywords:
                query_parts.extend(intent.keywords[:2])
            
            query = " ".join(query_parts) if query_parts else "popular music"
            market = "RU" if language == "ru" else "US"
            spotify_results = self.spotify_client.search_tracks(query, limit=limit, market=market)
        
        tracks = []
        for item in spotify_results:
            track = Track(
                spotify_id=item["spotify_id"],
                name=item["name"],
                artist=item["artist"],
                year=int(item["release_date"][:4]) if item.get("release_date") and len(item["release_date"]) >= 4 else None,
                language=language or "other",
                valence=features.get("valence", 0.5),
                energy=features.get("energy", 0.5),
                danceability=features.get("danceability", 0.5),
            )
            tracks.append(track)
        
        return tracks


_pipeline_instance: Optional[MusicRecommendationPipeline] = None


def get_pipeline() -> MusicRecommendationPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MusicRecommendationPipeline()
    return _pipeline_instance
