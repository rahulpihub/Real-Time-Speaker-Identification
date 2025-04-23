import os
import subprocess
import streamlit as st
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity

# ==== Setup ====
UPLOAD_DIR = 'uploads/'
os.makedirs(UPLOAD_DIR, exist_ok=True)
USER1_PATH = os.path.join(UPLOAD_DIR, 'user1_audio.wav')
USER2_PATH = os.path.join(UPLOAD_DIR, 'user2_audio.wav')
LIVE_PATH = os.path.join(UPLOAD_DIR, 'live_audio.wav')

encoder = VoiceEncoder()
user1_embedding = None
user2_embedding = None

st.title("🎙️ Real-Time Speaker Identification using Voice References")


# ==== Utilities ====
def convert_to_wav(src_path, dest_path):
    try:
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', src_path, '-ar', '16000', '-ac', '1', dest_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
            return {'error': f"FFmpeg failed. Log:\n{result.stderr.decode()}"}
        os.remove(src_path)
        return dest_path
    except Exception as e:
        return {'error': str(e)}


def get_speaker_embedding(wav_path):
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"File not found: {wav_path}")
    try:
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = np.mean(wav, axis=1)
        wav = preprocess_wav(wav, source_sr=sr)
        return encoder.embed_utterance(wav)
    except Exception as e:
        raise RuntimeError(f"Failed to process {wav_path}: {e}")


def identify_speakers_live(audio_path, ref_embeds):
    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav_preprocessed = preprocess_wav(wav, source_sr=sr)
    _, cont_embeds, _ = encoder.embed_utterance(wav_preprocessed, return_partials=True)

    segment_duration = len(wav_preprocessed) / sr / len(cont_embeds)
    diarization_result = []

    for i, embed in enumerate(cont_embeds):
        sims = [cosine_similarity([embed], [ref])[0][0] for ref in ref_embeds]
        speaker = f"User {np.argmax(sims) + 1}"
        diarization_result.append({
            "start": round(i * segment_duration, 2),
            "end": round((i + 1) * segment_duration, 2),
            "speaker": speaker
        })
    return diarization_result


# ==== Upload Reference for User 1 ====
uploaded_file_user1 = st.file_uploader("Upload reference for 🧑 User 1", type=["wav", "mp3"])
if uploaded_file_user1:
    raw_path = os.path.join(UPLOAD_DIR, uploaded_file_user1.name)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file_user1.getbuffer())

    result = convert_to_wav(raw_path, USER1_PATH)
    if isinstance(result, str):
        st.success("✅ User 1 audio uploaded and converted.")
        try:
            user1_embedding = get_speaker_embedding(USER1_PATH)
        except Exception as e:
            st.error(str(e))
    else:
        st.error(result['error'])


# ==== Upload Reference for User 2 ====
uploaded_file_user2 = st.file_uploader("Upload reference for 🧑 User 2", type=["wav", "mp3"])
if uploaded_file_user2:
    raw_path = os.path.join(UPLOAD_DIR, uploaded_file_user2.name)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file_user2.getbuffer())

    result = convert_to_wav(raw_path, USER2_PATH)
    if isinstance(result, str):
        st.success("✅ User 2 audio uploaded and converted.")
        try:
            user2_embedding = get_speaker_embedding(USER2_PATH)
        except Exception as e:
            st.error(str(e))
    else:
        st.error(result['error'])


# ==== Upload Live Audio for Speaker Diarization ====
if user1_embedding is not None and user2_embedding is not None:
    st.markdown("### 🎧 Upload a recorded conversation audio")
    uploaded_live = st.file_uploader("Upload live audio (wav/mp3)", type=["wav", "mp3"])

    if uploaded_live:
        raw_path = os.path.join(UPLOAD_DIR, uploaded_live.name)
        with open(raw_path, "wb") as f:
            f.write(uploaded_live.getbuffer())

        result = convert_to_wav(raw_path, LIVE_PATH)
        if isinstance(result, str):
            st.success("🎙️Audio uploaded and converted.")
            try:
                diarization = identify_speakers_live(LIVE_PATH, [user1_embedding, user2_embedding])
                st.write("🗣️ Speaker Diarization Result:")
                st.json(diarization)
            except Exception as e:
                st.error(f"Failed to diarize live audio: {e}")
        else:
            st.error(result['error'])
else:
    st.info("Please upload both User 1 and User 2 reference audios first.")
