import glob
import os
import yt_dlp
from faster_whisper import WhisperModel
import torch

# Load Whisper model globally to avoid reloading
whisper_model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu")

def download_audio(url, out_path="video.wav"):
    # Remove old .wav files
    for f in glob.glob("*.wav"):
        os.remove(f)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'video.%(ext)s',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    wav_files = glob.glob("*.wav")
    if not wav_files:
        return None
    latest_wav = max(wav_files, key=os.path.getctime)
    return latest_wav

def transcribe_with_faster_whisper(audio_path):
    segments, info = whisper_model.transcribe(audio_path, beam_size=1)
    text = " ".join([segment.text.strip() for segment in segments])
    return text
