import pandas as pd
import re

DATA_PATH = "data/data.csv"

_df = None


def load_data():
    global _df
    if _df is None:
        try:
            _df = pd.read_csv(DATA_PATH)
            _df["artists"] = _df["artists"].astype(str)
            _df["name"] = _df["name"].astype(str)
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {DATA_PATH} не найден.")
    return _df


def is_cyrillic(text):
    return bool(re.search("[а-яА-ЯёЁ]", text))


def recommend_tracks(llm_response: dict, top_k=5):
    df = load_data().copy()

    features_target = llm_response.get("features", {})
    filters = llm_response.get("filters", {})

    # Фильтр по годам
    if filters.get("year_start"):
        df = df[df["year"] >= filters["year_start"]]
    if filters.get("year_end"):
        df = df[df["year"] <= filters["year_end"]]

    # Фильтр по артисту (частичное совпадение, без учета регистра)
    if filters.get("artist"):
        artist_query = filters["artist"].lower()
        df = df[df["artists"].str.lower().str.contains(artist_query)]

    # Фильтр по языку (ищем кириллицу)
    if filters.get("language") == "ru":
        df = df[df["name"].apply(is_cyrillic) | df["artists"].apply(is_cyrillic)]

    # Если фильтры слишком строгие и ничего не осталось, вернем исходный датасет с предупреждением
    if df.empty:
        print("⚠️ Фильтры исключили все треки. Сбрасываем фильтры.")
        df = load_data().copy()


    # Защита от отсутствующих ключей
    v = features_target.get("valence", 0.5)
    e = features_target.get("energy", 0.5)
    d = features_target.get("danceability", 0.5)
    a = features_target.get("acousticness", 0.5)
    t = features_target.get("tempo", 120)

    df["dist"] = (
        (df["valence"] - v) ** 2
        + (df["energy"] - e) ** 2
        + (df["danceability"] - d) ** 2
        + (df["acousticness"] - a) ** 2
        + ((df["tempo"] - t) / 200) ** 2
    )

    recommendations = df.nsmallest(top_k, 'dist')
    
    result = []
    for _, row in recommendations.iterrows():
        track_dict = {
            'artists': row['artists'],
            'name': row['name'],
            'year': row['year'],
            'dist': row['dist'],
        }
        
        if 'id' in row:
            track_dict['id'] = row['id']
            track_dict['url'] = f"https://open.spotify.com/track/{row['id']}"
        else:
            track_dict['url'] = "#"
            
        result.append(track_dict)
        
    return result
