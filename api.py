from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transcribe_task import download_audio, transcribe_with_faster_whisper
from summary_task import SummaryModel, SummaryTemplate, SummaryGenerate, cohere_tokenizer, cohere_model

app = FastAPI()

class VideoInput(BaseModel):
    url: str

@app.post("/summarize_video")
def summarize_video(input: VideoInput):
    try:
        audio_path = download_audio(input.url)
        if not audio_path:
            raise HTTPException(status_code=500, detail="Download failed.")
        text_article = transcribe_with_faster_whisper(audio_path)
        template = SummaryTemplate(text_article, SummaryModel)
        summary = SummaryGenerate(template, cohere_tokenizer, cohere_model, max_new_tokens=len(text_article) + 100)
        return {"summary": summary["Summary"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
