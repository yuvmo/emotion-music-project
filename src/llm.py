import json
from src.utils import get_llm
from langchain_core.messages import HumanMessage

def get_music_params(text: str, emotion_label: str) -> dict:
    llm = get_llm()
    
    prompt_text = f"""
    Ты музыкальный ассистент. Пользователь произнес фразу с эмоцией "{emotion_label}".
    Текст: "{text}"
    
    Твоя задача:
    1. Определить параметры музыки (valence, energy, etc).
    2. Извлечь жесткие фильтры, если они есть в тексте (год, артист, язык).

    Если пользователь просит "русские" треки, ставь "language": "ru".
    Если пользователь называет десятилетие (например, "2010-е"), ставь year_start=2010, year_end=2019.
    Если пользователь называет артиста, запиши его имя.

    Верни JSON строго следующей структуры:
    {{
        "features": {{
            "valence": 0.5,
            "energy": 0.5,
            "danceability": 0.5,
            "acousticness": 0.1,
            "tempo": 120
        }},
        "filters": {{
            "year_start": null,  
            "year_end": null,
            "artist": null,      
            "language": null     
        }}
    }}
    
    Пример 1: "хочу грустный русский рок 2005 года" -> filters: {{ "year_start": 2005, "year_end": 2005, "artist": null, "language": "ru" }}
    Пример 2: "веселые песни" -> filters: {{ "year_start": null, ... }}
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        content = response.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Ошибка LLM: {e}")
        return {
            "features": {"valence": 0.5, "energy": 0.5, "danceability": 0.5, "acousticness": 0.0, "tempo": 120},
            "filters": {"year_start": None, "year_end": None, "artist": None, "language": None}
        }