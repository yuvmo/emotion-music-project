import json
import logging
from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage

from src.utils import get_llm
from src.intent.extractor import UserIntent
from .prompts import PromptBuilder

logger = logging.getLogger(__name__)


class GigaChatService:
    def __init__(self):
        self._llm = None
        self.prompt_builder = PromptBuilder()
    
    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    def _parse_json_response(self, content: str) -> Optional[dict]:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None
    
    def analyze_music_request(self, intent: UserIntent) -> Dict[str, Any]:
        llm = self._get_llm()
        prompt = self.prompt_builder.build_music_analysis_prompt(intent)
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_json_response(response.content)
            
            if result:
                return result
            
        except Exception as e:
            logger.error(f"GigaChat error: {e}")
        
        return self._get_fallback_params(intent)
    
    def _get_fallback_params(self, intent: UserIntent) -> Dict[str, Any]:
        from src.audio.emotion import EMOTION_MUSIC_PROFILES
        
        emotion = intent.audio_emotion or "neutral"
        profile = EMOTION_MUSIC_PROFILES.get(emotion, EMOTION_MUSIC_PROFILES["neutral"])
        
        valence = sum(profile["valence"]) / 2
        energy = sum(profile["energy"]) / 2
        danceability = sum(profile["danceability"]) / 2
        tempo = sum(profile["tempo"]) / 2
        
        return {
            "mood_interpretation": f"Настроение: {profile['description']}",
            "features": {
                "valence": valence,
                "energy": energy,
                "danceability": danceability,
                "acousticness": 0.3,
                "tempo": tempo
            },
            "filters": {
                "genres": intent.genres if intent.genres else None,
                "language": intent.language,
                "year_start": None,
                "year_end": None,
                "artist": intent.artist
            },
            "explanation": f"Fallback based on emotion '{emotion}'"
        }
    
    def generate_response(self, tracks: list, intent: UserIntent, mood_interpretation: str) -> str:
        llm = self._get_llm()
        prompt = self.prompt_builder.build_recommendation_response_prompt(
            tracks, intent, mood_interpretation
        )
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return self._get_fallback_response(intent)
    
    def _get_fallback_response(self, intent: UserIntent) -> str:
        emotion = intent.audio_emotion
        
        responses = {
            "happy": "Вижу отличное настроение! Подобрал треки, чтобы продлить позитив!",
            "sad": "Понимаю тебя. Вот музыка, которая поможет пережить этот момент.",
            "angry": "Чувствую энергию! Эти треки помогут выплеснуть эмоции.",
            "energetic": "Ловлю драйв! Вот музыка для заряда!",
            "calm": "Подобрал спокойные треки для расслабления.",
            "neutral": "Вот подборка под твое настроение!"
        }
        
        return responses.get(emotion, responses["neutral"])
    
    def generate_clarification(self, intent: UserIntent) -> str:
        llm = self._get_llm()
        prompt = self.prompt_builder.build_clarification_prompt(intent)
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Clarification generation error: {e}")
            return "Не совсем понял, какую музыку ты хочешь. Можешь сказать подробнее — какое у тебя настроение или какой жанр?"


_service_instance: Optional[GigaChatService] = None


def get_gigachat_service() -> GigaChatService:
    global _service_instance
    if _service_instance is None:
        _service_instance = GigaChatService()
    return _service_instance
