from typing import List, Dict

from .models.text_emotion_model import get_text_emotion_model


def detect_text_emotion(
    text: str,
    top_k: int = 3,
) -> List[Dict[str, float]]:
    """
    По тексту возвращает top_k эмоций.

    Возвращает список словарей:
    [
      {"label": "joy", "score": 0.82},
      {"label": "sadness", "score": 0.10},
      ...
    ]
    """
    model = get_text_emotion_model()
    predictions = model.predict(texts=text, top_k=top_k)
    return predictions[0]
