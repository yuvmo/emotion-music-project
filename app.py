import streamlit as st
import os
from src.stt import transcribe
from src.text_emotion import detect_text_emotion
from src.llm import get_music_params
from src.music import recommend_tracks

st.set_page_config(page_title="Music Emotion AI", page_icon="🎵", layout="wide")

st.title("🎵 Music Emotion AI")
st.markdown("Скажи, как ты себя чувствуешь, и я подберу музыку.")

with st.sidebar:
    st.header("Настройки")
    input_method = st.radio("Ввод аудио:", ["Загрузить файл", "Запись микрофона"])
    top_k = st.slider("Количество треков", 1, 10, 5)


def process_audio(audio_path):
    with st.status("Обработка...", expanded=True) as status:
        st.write("🎙️ Распознавание речи...")
        try:
            text = transcribe(audio_path)
            st.success(f"Текст: {text}")
        except Exception as e:
            st.error(f"Ошибка STT: {e}")
            return

        st.write("🤔 Анализ эмоций...")
        try:
            emotions = detect_text_emotion(text)
            top_emotion = emotions[0]

            cols = st.columns(len(emotions))
            for i, emo in enumerate(emotions):
                cols[i].metric(label=emo["label"], value=f"{emo['score']:.2f}")

        except Exception as e:
            st.error(f"Ошибка Emotion: {e}")
            return

        st.write("🎛️ Генерация параметров и фильтров...")
        try:
            llm_response = get_music_params(text, top_emotion["label"])

            st.json(llm_response, expanded=False)

            params = llm_response["features"]

        except Exception as e:
            st.error(f"Ошибка LLM: {e}")
            return

        st.write("🎵 Подбор треков...")
        tracks = recommend_tracks(llm_response, top_k=top_k)
        status.update(label="Готово!", state="complete", expanded=False)
        return tracks, params, text, top_emotion


audio_file = None

if input_method == "Загрузить файл":
    uploaded_file = st.file_uploader(
        "Выберите аудио (mp3, wav)", type=["mp3", "wav", "ogg"]
    )
    if uploaded_file is not None:
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio_file = "temp_audio.mp3"
        st.audio(audio_file)

elif input_method == "Запись микрофона":
    from streamlit_mic_recorder import mic_recorder

    audio_data = mic_recorder(
        start_prompt="Начать запись", stop_prompt="Стоп", key="recorder"
    )

    if audio_data:
        with open("temp_recorded.mp3", "wb") as f:
            f.write(audio_data["bytes"])
        audio_file = "temp_recorded.mp3"
        st.audio(audio_data["bytes"])

if audio_file and st.button("Подобрать музыку", type="primary"):
    result = process_audio(audio_file)

    if result and result[0] is not None:
        tracks, params, text, top_emotion = result

        st.divider()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Анализ")
            st.info(f"🎭 Эмоция: **{top_emotion['label']}**")

            st.caption("Параметры музыки:")
            st.progress(
                params.get("valence", 0),
                text=f"Positivity: {params.get('valence'):.2f}",
            )
            st.progress(
                params.get("energy", 0), text=f"Energy: {params.get('energy'):.2f}"
            )
            st.progress(
                params.get("danceability", 0),
                text=f"Danceability: {params.get('danceability'):.2f}",
            )

        with col2:
            st.subheader("Рекомендации")
            for t in tracks:
                with st.expander(f"🎶 {t['artists']} - {t['name']}"):
                    st.write(f"Год: {t['year']}")
                    st.write(f"Соответствие: {t['dist']:.4f}")
                    st.markdown(f"[🎧 Слушать в Spotify]({t['url']})")

if os.path.exists("temp_audio.mp3"):
    os.remove("temp_audio.mp3")
