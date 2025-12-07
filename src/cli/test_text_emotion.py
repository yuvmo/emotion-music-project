from src.text_module.service import detect_text_emotion


def main():
    examples = [
        # Английский
        "I am so happy today, everything is going great!",
        "I'm really scared about tomorrow.",
        "I'm angry about what happened at work.",

        # Русский
        "Я очень рад сегодня, всё идёт отлично!",
        "Мне очень грустно и одиноко.",
        "Я злюсь из-за того, что произошло на работе.",
        "Мне страшно за будущее.",
    ]

    for text in examples:
        emotions = detect_text_emotion(text, top_k=4)
        print("=" * 80)
        print(f"TEXT: {text}")
        for e in emotions:
            label = e["label"]
            score = round(e["score"], 3)
            print(f"  {label:10s} -> {score}")


if __name__ == "__main__":
    main()
