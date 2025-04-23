from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from openai import OpenAI
import os
import tempfile

app = FastAPI(title="Chatbot API", description="An API to access an OpenAI chat.", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set your OpenAI API key as an environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Question(BaseModel):
    context: str
    text: str

class Answer(BaseModel):
    response: str

@app.post("/ask", response_model=Answer, summary="Ask the AI a question.", description="Sends a text question to OpenAI and returns the generated response.")
async def ask_holocron(question: Question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": question.context},
                {"role": "user", "content": question.text}
            ]
        )
        ai_response = response.choices[0].message.content
        return {"response": ai_response}
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/talk", summary="Talk to the AI", description="Send a voice message and get a voice response with contextual behavior.")
async def talk_to_ai(file: UploadFile = File(...), context: str = Form(default="You are a helpful assistant.")):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcribe audio
        audio_file = open(tmp_path, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        print("TRANSCRIPT:", transcript)
        print("CONTEXT:", context)

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Empty transcription. Check audio clarity.")

        # Call GPT with context + transcript
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": transcript}
            ]
        )
        reply_text = completion.choices[0].message.content
        print("GPT RESPONSE:", reply_text)

        if not reply_text.strip():
            raise HTTPException(status_code=400, detail="Empty GPT response. Check prompt and context.")

        # Text-to-Speech
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=reply_text
        )

        return StreamingResponse(speech_response.iter_bytes(), media_type="audio/mpeg")

    except Exception as e:
        print("ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", summary="Check API health", description="Returns a simple message to check if the API is running.")
async def health_check():
    return {"status": "ok"}
