import pandas as pd
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
METRICS_FILE = DATA_DIR / "metrics.csv"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"


def load_data():
    metrics_df = None
    feedback_df = None
    
    if METRICS_FILE.exists():
        metrics_df = pd.read_csv(METRICS_FILE)
        print(f"Загружено {len(metrics_df)} записей метрик")
    else:
        print("Файл metrics.csv не найден")
    
    if FEEDBACK_FILE.exists():
        feedback_df = pd.read_csv(FEEDBACK_FILE)
        print(f"Загружено {len(feedback_df)} отзывов")
    else:
        print("Файл feedback.csv не найден")
    
    return metrics_df, feedback_df


def analyze_metrics(df):
    if df is None or df.empty:
        print("\nНет данных для анализа метрик")
        return
    
    print("\n" + "=" * 60)
    print("МЕТРИКИ ПАЙПЛАЙНА")
    print("=" * 60)
    
    total = len(df)
    successful = df["success"].sum()
    print(f"\nОбщая статистика:")
    print(f"  Всего запросов: {total}")
    print(f"  Успешных: {successful} ({successful/total*100:.1f}%)")
    
    if "processing_time_sec" in df.columns:
        print(f"\nВремя обработки (E2E Latency):")
        print(f"  Среднее: {df['processing_time_sec'].mean():.2f} сек")
        print(f"  Медиана: {df['processing_time_sec'].median():.2f} сек")
        print(f"  Макс: {df['processing_time_sec'].max():.2f} сек")
        print(f"  Мин: {df['processing_time_sec'].min():.2f} сек")
    
    if "audio_duration_sec" in df.columns:
        print(f"\nДлительность аудио:")
        print(f"  Средняя: {df['audio_duration_sec'].mean():.2f} сек")
        print(f"  Медиана: {df['audio_duration_sec'].median():.2f} сек")
    
    if "audio_valid" in df.columns:
        valid = df["audio_valid"].sum()
        print(f"\nВалидация аудио:")
        print(f"  Валидных: {valid} ({valid/total*100:.1f}%)")
        print(f"  Отклонено: {total - valid} ({(total-valid)/total*100:.1f}%)")
        
        if "validation_error" in df.columns:
            errors = df[df["audio_valid"] == False]["validation_error"].value_counts()
            if not errors.empty:
                print(f"  Причины отклонения:")
                for reason, count in errors.items():
                    print(f"    - {reason}: {count}")
    
    if "emotion" in df.columns:
        print(f"\nРаспределение эмоций:")
        emotions = df["emotion"].value_counts()
        for emotion, count in emotions.items():
            if emotion and str(emotion) != "nan":
                print(f"  {emotion}: {count} ({count/total*100:.1f}%)")
    
    if "intents_count" in df.columns:
        with_intents = (df["intents_count"] > 0).sum()
        print(f"\nИзвлечение интентов:")
        print(f"  С интентами: {with_intents} ({with_intents/total*100:.1f}%)")
        print(f"  Без интентов: {total - with_intents}")
    
    if "intents_genre" in df.columns:
        genres = df["intents_genre"].dropna()
        genres = genres[genres != ""]
        if not genres.empty:
            all_genres = []
            for g in genres:
                all_genres.extend(g.split(","))
            genre_counts = pd.Series(all_genres).value_counts()
            print(f"\n  Топ жанры:")
            for genre, count in genre_counts.head(5).items():
                print(f"    - {genre}: {count}")
    
    if "intents_language" in df.columns:
        langs = df["intents_language"].value_counts()
        print(f"\n  Языки:")
        for lang, count in langs.items():
            if lang and str(lang) != "nan":
                print(f"    - {lang}: {count}")
    
    if "llm_success" in df.columns:
        llm_ok = df["llm_success"].sum()
        print(f"\nGigaChat анализ:")
        print(f"  Успешных: {llm_ok} ({llm_ok/total*100:.1f}%)")
        if "llm_time_sec" in df.columns:
            print(f"  Среднее время: {df['llm_time_sec'].mean():.2f} сек")
    
    if "tracks_found" in df.columns:
        with_tracks = (df["tracks_found"] > 0).sum()
        print(f"\nРекомендации (Hit Rate):")
        print(f"  С треками: {with_tracks} ({with_tracks/total*100:.1f}%)")
        print(f"  Без треков: {total - with_tracks}")
        print(f"  Среднее кол-во треков: {df['tracks_found'].mean():.1f}")
    
    if "tracks_from_dataset" in df.columns and "tracks_from_spotify" in df.columns:
        from_ds = df["tracks_from_dataset"].sum()
        from_sp = df["tracks_from_spotify"].sum()
        total_tracks = from_ds + from_sp
        if total_tracks > 0:
            print(f"\nИсточники треков:")
            print(f"  Из датасета: {from_ds} ({from_ds/total_tracks*100:.1f}%)")
            print(f"  Из Spotify: {from_sp} ({from_sp/total_tracks*100:.1f}%)")
    
    if "target_valence" in df.columns:
        print(f"\nСредние целевые параметры:")
        print(f"  Valence: {df['target_valence'].mean():.3f}")
        print(f"  Energy: {df['target_energy'].mean():.3f}")
        print(f"  Danceability: {df['target_danceability'].mean():.3f}")
        print(f"  Tempo: {df['target_tempo'].mean():.1f}")


def analyze_feedback(df):
    if df is None or df.empty:
        print("\nНет данных для анализа отзывов")
        return
    
    print("\n" + "=" * 60)
    print("ОТЗЫВЫ ПОЛЬЗОВАТЕЛЕЙ")
    print("=" * 60)
    
    total = len(df)
    
    if "feedback" in df.columns:
        feedback_counts = df["feedback"].value_counts()
        good = feedback_counts.get("good", 0)
        bad = feedback_counts.get("bad", 0)
        
        print(f"\nUser Satisfaction:")
        print(f"  Всего отзывов: {total}")
        print(f"  Положительных: {good} ({good/total*100:.1f}%)")
        print(f"  Отрицательных: {bad} ({bad/total*100:.1f}%)")
        
        if good + bad > 0:
            satisfaction_rate = good / (good + bad) * 100
            print(f"\n  Satisfaction Rate: {satisfaction_rate:.1f}%")
    
    if "emotion" in df.columns:
        print(f"\nОтзывы по эмоциям:")
        grouped = df.groupby("emotion")["feedback"].value_counts().unstack(fill_value=0)
        if not grouped.empty:
            for emotion in grouped.index:
                row = grouped.loc[emotion]
                good = row.get("good", 0)
                bad = row.get("bad", 0)
                total_em = good + bad
                if total_em > 0:
                    rate = good / total_em * 100
                    print(f"  {emotion}: {good}/{total_em} положительных ({rate:.0f}%)")


def main():
    print("Анализ метрик Music Emotion Bot")
    print("-" * 40)
    
    metrics_df, feedback_df = load_data()
    
    analyze_metrics(metrics_df)
    analyze_feedback(feedback_df)
    
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 60)
    
    if metrics_df is not None and not metrics_df.empty:
        if "processing_time_sec" in metrics_df.columns:
            avg_time = metrics_df["processing_time_sec"].mean()
            if avg_time > 10:
                print("- Высокое время обработки. Рассмотреть оптимизацию моделей.")
        
        if "tracks_found" in metrics_df.columns:
            hit_rate = (metrics_df["tracks_found"] > 0).mean()
            if hit_rate < 0.8:
                print("- Низкий Hit Rate. Расширить датасет или ослабить фильтры.")
    
    if feedback_df is not None and not feedback_df.empty:
        if "feedback" in feedback_df.columns:
            good = (feedback_df["feedback"] == "good").sum()
            total = len(feedback_df)
            if good / total < 0.7:
                print("- Низкая удовлетворенность. Улучшить промпты GigaChat.")


if __name__ == "__main__":
    main()

