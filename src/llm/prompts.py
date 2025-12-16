from typing import Optional, List, Dict, Any
from src.audio.emotion import EMOTION_MUSIC_PROFILES
from src.intent.extractor import UserIntent


class PromptBuilder:
    AVAILABLE_GENRES = [
        "pop", "dance", "electronic", "indie",
        "hip_hop", "rap", "trap", "rnb",
        "rock", "metal", "punk", "alternative",
        "classical", "instrumental", "ambient", "jazz",
        "folk", "latin", "soundtrack", "blues"
    ]
    
    AVAILABLE_LANGUAGES = ["ru", "en", "instrumental", "other"]
    
    @staticmethod
    def build_music_analysis_prompt(intent: UserIntent) -> str:
        emotion_context = ""
        if intent.audio_emotion:
            profile = EMOTION_MUSIC_PROFILES.get(intent.audio_emotion, {})
            emotion_desc = profile.get("description", "нейтральная")
            confidence = intent.audio_emotion_confidence or 0
            emotion_context = f"""
По голосу пользователя определена эмоция: {intent.audio_emotion} ({emotion_desc})
Уверенность: {confidence:.0%}
"""
        
        text_context = f'Пользователь сказал: "{intent.transcript}"' if intent.transcript else ""
        
        preferences_context = ""
        if intent.genres:
            preferences_context += f"\nИз текста извлечены жанры: {', '.join(intent.genres)}"
        if intent.language:
            lang_map = {"ru": "русский", "en": "английский", "instrumental": "инструментальная музыка"}
            preferences_context += f"\nПредпочтение языка: {lang_map.get(intent.language, intent.language)}"
        if intent.mood_keywords:
            preferences_context += f"\nКлючевые слова настроения: {', '.join(intent.mood_keywords)}"
        
        prompt = f"""Ты умный музыкальный ассистент. Твоя задача — понять настроение пользователя и подобрать идеальные параметры музыки.

КОНТЕКСТ:
{emotion_context}
{text_context}
{preferences_context}

ДОСТУПНЫЕ ЖАНРЫ В БАЗЕ: {', '.join(PromptBuilder.AVAILABLE_GENRES)}
ДОСТУПНЫЕ ЯЗЫКИ: ru (русский), en (английский), instrumental (без слов), other (любой)

ТВОЯ ЗАДАЧА:
1. Проанализируй эмоциональное состояние пользователя (по голосу + по словам)
2. Определи, какую музыку он хочет: поднять настроение, успокоиться, зарядиться энергией, погрустить вместе?
3. Подбери музыкальные параметры Spotify:
   - valence (0-1): позитивность. 0 = грустная, 1 = веселая
   - energy (0-1): энергичность. 0 = спокойная, 1 = интенсивная
   - danceability (0-1): танцевальность
   - acousticness (0-1): акустичность. 0 = электронная, 1 = живые инструменты
   - tempo (60-200): темп в BPM

4. Уточни фильтры:
   - genres: список жанров (1-3 штуки, из доступных)
   - language: язык треков (ru/en/instrumental/other или null для любого)
   - year_start, year_end: если пользователь упомянул период (иначе null)
   - artist: имя исполнителя, если пользователь упомянул конкретного артиста (иначе null)

5. РАСПОЗНАВАНИЕ АРТИСТА (ВАЖНО!):
   - Если пользователь упомянул исполнителя — определи его правильное имя
   - Whisper может ошибаться в транскрипции имён: "OG Booda", "О.Г. Будо", "Оджибуда" = "OG Buda"
   - "Скриптонит", "скрипт", "scriptonit" = "Скриптонит"
   - "Монеточка", "монета" = "Монеточка"  
   - "Оксимирон", "окси", "oxxxymiron" = "Oxxxymiron"
   - "Фейс", "face" = "Face"
   - Если не уверен — верни наиболее вероятное имя артиста
   - Имя пиши так, как принято в Spotify (на языке оригинала)

ВАЖНО:
- Если пользователь грустит, НЕ всегда нужна грустная музыка — иногда нужно поднять настроение
- Если эмоция angry, но текст нейтральный — доверяй голосу больше
- Если жанры уже извлечены, используй их, но можешь дополнить
- Параметры должны соответствовать комбинации эмоции + жанра

Верни ТОЛЬКО JSON (без markdown, без пояснений):
{{
    "mood_interpretation": "краткое описание того, что хочет пользователь",
    "features": {{
        "valence": 0.5,
        "energy": 0.5,
        "danceability": 0.5,
        "acousticness": 0.3,
        "tempo": 120
    }},
    "filters": {{
        "genres": ["pop"],
        "language": null,
        "year_start": null,
        "year_end": null,
        "artist": null
    }},
    "explanation": "почему такой выбор параметров"
}}"""
        
        return prompt
    
    @staticmethod
    def build_recommendation_response_prompt(
        tracks: List[Dict[str, Any]], 
        intent: UserIntent,
        mood_interpretation: str
    ) -> str:
        tracks_list = "\n".join([
            f"- {t.get('artist', 'Unknown')} — {t.get('name', 'Unknown')}"
            for t in tracks[:5]
        ])
        
        emotion_text = ""
        if intent.audio_emotion:
            emotion_text = f"Эмоция в голосе: {intent.audio_emotion}"
        
        prompt = f"""Ты дружелюбный музыкальный бот. Пользователь попросил подобрать музыку.

Что он сказал: "{intent.transcript}"
{emotion_text}
Мы поняли его запрос так: {mood_interpretation}

Мы подобрали эти треки:
{tracks_list}

Напиши короткий (2-3 предложения) дружелюбный ответ:
1. Покажи, что ты понял его настроение
2. Объясни, почему эти треки подойдут
3. Добавь эмодзи для выразительности

НЕ перечисляй треки — они будут показаны отдельно.
Будь живым и эмпатичным, не формальным."""
        
        return prompt
    
    @staticmethod
    def build_clarification_prompt(intent: UserIntent) -> str:
        prompt = f"""Пользователь отправил голосовое сообщение, но его запрос неясен.

Текст: "{intent.transcript}"
Эмоция в голосе: {intent.audio_emotion or "не определена"}

Мы не смогли понять:
- Какую музыку он хочет
- Какое у него настроение

Напиши короткий дружелюбный уточняющий вопрос (1-2 предложения).
Предложи варианты, например:
- Хотите музыку под настроение или конкретный жанр?
- Может, что-то энергичное или спокойное?

Используй эмодзи, будь дружелюбным."""
        
        return prompt
