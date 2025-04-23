# 🎙️ Real-Time Speaker Identification with Voice References

This is a Streamlit web application for **real-time speaker identification** using voice reference samples. By uploading two reference audios (e.g., from two different users), the app can analyze a conversation and determine **who is speaking when** using **speaker diarization**.

---

## 🚀 Features

- ✅ Upload voice reference audio for **User 1** and **User 2**
- 🎧 Upload a **Recorded conversation audio**
- 🧠 Uses **Resemblyzer** for speaker embeddings
- 📊 Performs **speaker diarization** (segment-wise speaker prediction)
- 📈 Identifies who is speaking at each moment using **cosine similarity**

---

## 📂 How It Works

1. **Upload Reference Audios**  
   Upload clean voice samples (WAV/MP3) for **User 1** and **User 2**.  
   The app:
   - Converts them to mono WAV format (16 kHz)
   - Extracts a speaker **embedding** using `VoiceEncoder`

2. **Upload Live Audio**  
   Upload a live recording (conversation).  
   The app:
   - Converts the file
   - Splits it into short segments
   - Extracts embeddings for each segment
   - Compares each segment to the reference voices using **cosine similarity**
   - Tags each segment as "User 1" or "User 2"

3. **Output**  
   A detailed, time-based diarization result showing:
   - Start and end time of each speaker segment
   - Which user was speaking

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – for UI
- **Resemblyzer** – for speaker voice embeddings
- **FFmpeg** – for audio conversion
- **Scikit-learn** – for cosine similarity
- **SoundFile & NumPy** – for audio handling

---

## 🔧 Installation

```bash
# Clone this repo
git clone https://github.com/yourusername/speaker-diarization-app.git
cd speaker-diarization-app

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
