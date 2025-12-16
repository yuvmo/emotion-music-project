import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UserIntent:
    audio_emotion: Optional[str] = None
    audio_emotion_confidence: Optional[float] = None
    language: Optional[str] = None
    genres: List[str] = field(default_factory=list)
    mood_keywords: List[str] = field(default_factory=list)
    play_intent: bool = False
    transcript: str = ""
    artist: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "audio_emotion": self.audio_emotion,
            "audio_emotion_confidence": self.audio_emotion_confidence,
            "language": self.language,
            "genres": self.genres,
            "mood_keywords": self.mood_keywords,
            "play_intent": self.play_intent,
            "transcript": self.transcript,
            "artist": self.artist,
            "keywords": self.keywords,
        }
    
    def has_preferences(self) -> bool:
        return bool(self.language or self.genres or self.mood_keywords or self.artist)


GENRE_KEYWORDS = {
    "pop": {"pop", "поп", "попс"},
    "dance": {"dance", "танц", "дэнс"},
    "electronic": {"electronic", "edm", "house", "techno", "trance", "электрон", "хаус", "техно", "транс"},
    "indie": {"indie", "инди"},
    "hip_hop": {"hip", "hop", "хип", "хоп"},
    "rap": {"rap", "рэп", "реп", "рэпчик"},
    "trap": {"trap", "трэп", "треп"},
    "rnb": {"rnb", "r&b", "рнб", "ритм"},
    "rock": {"rock", "рок"},
    "metal": {"metal", "метал", "металл"},
    "punk": {"punk", "панк"},
    "alternative": {"alternative", "альтернатив", "альт"},
    "classical": {"classical", "классик", "symphon", "orchestr", "оркест", "симфон"},
    "instrumental": {"instrumental", "инструмент", "piano", "пиан", "фортепиано"},
    "ambient": {"ambient", "эмбиент", "амбиент"},
    "jazz": {"jazz", "джаз"},
    "folk": {"folk", "фолк", "народн"},
    "latin": {"latin", "латино", "регетон", "reggaeton"},
    "soundtrack": {"soundtrack", "саундтрек", "score", "кино", "фильм"},
    "blues": {"blues", "блюз"},
}

LANGUAGE_KEYWORDS = {
    "ru": ["русск", "российск", "по-русск", "отечествен", "наш"],
    "en": ["английск", "по-английск", "english", "англоязычн", "зарубежн", "иностран"],
    "instrumental": ["инструментал", "без слов", "без вокал", "фонов", "саундтрек", "без текст"],
}

MOOD_KEYWORDS = {
    "happy": ["весел", "радост", "позитив", "жизнерадост", "счастл", "хорош", "отличн"],
    "sad": ["грустн", "печальн", "тоскл", "меланхол", "плох", "одинок", "горе"],
    "energetic": ["энергичн", "бодр", "драйв", "активн", "мощн", "зажигательн"],
    "calm": ["спокойн", "расслаб", "тих", "умиротвор", "мягк", "нежн", "уютн"],
    "angry": ["злост", "агрессив", "ярост", "бешен", "дик", "жёстк", "жестк"],
    "romantic": ["романтичн", "любов", "нежн", "чувствен", "лиричн"],
    "nostalgic": ["ностальг", "старых", "прошл", "ретро", "винтаж"],
    "party": ["вечеринк", "тусовк", "клуб", "танцпол", "пати", "движ"],
}

PLAY_KEYWORDS = {
    "включи", "поставь", "запусти", "проиграй", "воспроизведи",
    "хочу", "давай", "дай", "найди", "подбери", "порекомендуй",
    "play", "start", "put"
}


class IntentExtractor:
    def __init__(self):
        self.genre_keywords = GENRE_KEYWORDS
        self.language_keywords = LANGUAGE_KEYWORDS
        self.mood_keywords = MOOD_KEYWORDS
        self.play_keywords = PLAY_KEYWORDS
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())
    
    def extract_genres(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        found_genres = set()
        
        for genre, keywords in self.genre_keywords.items():
            for token in tokens:
                for kw in keywords:
                    if token.startswith(kw) or kw in token:
                        found_genres.add(genre)
                        break
        
        return list(found_genres)
    
    def extract_language(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        
        for kw in self.language_keywords["instrumental"]:
            if kw in text_lower:
                return "instrumental"
        
        for kw in self.language_keywords["ru"]:
            if kw in text_lower:
                return "ru"
        
        for kw in self.language_keywords["en"]:
            if kw in text_lower:
                return "en"
        
        return None
    
    def extract_mood(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_moods = []
        
        for mood, keywords in self.mood_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    found_moods.append(mood)
                    break
        
        return found_moods
    
    def extract_play_intent(self, text: str) -> bool:
        tokens = self.tokenize(text)
        return any(token in self.play_keywords for token in tokens)
    
    def extract_keywords(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        stop_words = {
            "я", "ты", "мы", "вы", "он", "она", "они", "оно", "что", "как", 
            "это", "тот", "такой", "такая", "такие", "мне", "мой", "моя", "мои",
            "тебе", "твой", "хочу", "хочется", "нужно", "нужна", "можно", "можешь",
            "давай", "дай", "включи", "поставь", "послушать", "слушать", "музык",
            "песн", "трек", "что", "нибудь", "какой", "какую", "какие", "очень",
            "немного", "чуть", "просто", "сейчас", "сегодня", "вчера", "потом",
            "ещё", "еще", "только", "уже", "тоже", "также", "или", "либо", "а", "и",
            "но", "да", "нет", "не", "под", "на", "в", "к", "от", "для", "с", "по",
        }
        
        keywords = []
        for token in tokens:
            if len(token) >= 3 and token not in stop_words:
                keywords.append(token)
        
        return keywords[:5]
    
    def extract(
        self, 
        text: str,
        audio_emotion: Optional[str] = None,
        audio_emotion_confidence: Optional[float] = None
    ) -> UserIntent:
        if not text or not text.strip():
            return UserIntent(
                audio_emotion=audio_emotion,
                audio_emotion_confidence=audio_emotion_confidence,
                transcript=""
            )
        
        return UserIntent(
            audio_emotion=audio_emotion,
            audio_emotion_confidence=audio_emotion_confidence,
            language=self.extract_language(text),
            genres=self.extract_genres(text),
            mood_keywords=self.extract_mood(text),
            play_intent=self.extract_play_intent(text),
            transcript=text.strip(),
            artist=None,
            keywords=self.extract_keywords(text)
        )


_extractor_instance: Optional[IntentExtractor] = None


def get_intent_extractor() -> IntentExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = IntentExtractor()
    return _extractor_instance


def extract_user_intent(
    text: str,
    audio_emotion: Optional[str] = None,
    audio_emotion_confidence: Optional[float] = None
) -> UserIntent:
    extractor = get_intent_extractor()
    return extractor.extract(text, audio_emotion, audio_emotion_confidence)
