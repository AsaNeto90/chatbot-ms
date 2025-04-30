from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from openai import OpenAI
import os
import tempfile
from typing import Dict, List, Optional

# --- Configuration ---
MAX_USER_MESSAGES = 3 # Max number of user messages per session before cutoff
SESSION_LIMIT_MESSAGE = ( # Message to send when the limit is reached
    "Our current conversation has reached its planned length. "
    "To explore a new topic or continue our discussion further, "
    "please begin a new session."
)
# --- End Configuration ---

app = FastAPI(title="Chatbot API", description="An API to access an OpenAI chat with session persistence and message limits.", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set your OpenAI API key as an environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Session Management ---
session_histories: Dict[str, List[Dict[str, str]]] = {}
# --- End Session Management ---

class Question(BaseModel):
    context: str # System prompt / initial context for the session
    text: str

class Answer(BaseModel):
    response: str
    sessionId: str
    limit_reached: Optional[bool] = False # Indicate if the limit was hit

# --- Helper Function ---
def count_user_messages(history: List[Dict[str, str]]) -> int:
    """Counts the number of messages with role 'user' in the history."""
    return sum(1 for msg in history if msg.get("role") == "user")
# --- End Helper Function ---


@app.post("/ask", response_model=Answer, summary="Ask the AI a question with session persistence and limit.", description="Sends text, maintains context via sessionId, returns response. Stops after MAX_USER_MESSAGES.")
async def ask_holocron(question: Question, sessionId: str = Query(...)):
    # Retrieve history or initialize if new session
    if sessionId not in session_histories:
        session_histories[sessionId] = [{"role": "system", "content": question.context}]
    
    history = session_histories[sessionId]
    
    # --- Check Message Limit ---
    user_message_count = count_user_messages(history)
    if user_message_count >= MAX_USER_MESSAGES:
        print(f"Session {sessionId} reached message limit ({user_message_count}/{MAX_USER_MESSAGES}).")
        # Return the predefined message without calling the AI
        return Answer(
            response=SESSION_LIMIT_MESSAGE, 
            sessionId=sessionId,
            limit_reached=True 
        )
    # --- End Check Message Limit ---
        
    # Limit not reached, proceed normally
    history.append({"role": "user", "content": question.text})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history 
        )
        ai_response = response.choices[0].message.content
        
        if ai_response:
             history.append({"role": "assistant", "content": ai_response})
        
        session_histories[sessionId] = history
        
        return Answer(response=ai_response, sessionId=sessionId)
        
    except openai.error.OpenAIError as e:
        if history and history[-1]["role"] == "user":
             history.pop() # Rollback user message on error
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        if history and history[-1]["role"] == "user":
             history.pop() # Rollback user message on error
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/talk", summary="Talk to the AI with session persistence and limit", description="Send voice, maintain context via sessionId, get voice response. Stops after MAX_USER_MESSAGES.")
async def talk_to_ai(
    sessionId: str = Form(...),
    file: UploadFile = File(...),
    context: str = Form(default="You are a helpful assistant.") # Initial context for new sessions
):
    # Retrieve history or initialize if new session
    if sessionId not in session_histories:
        session_histories[sessionId] = [{"role": "system", "content": context}]
        
    history = session_histories[sessionId]
    tmp_path = None # Initialize tmp_path

    try:
        # --- Check Message Limit ---
        user_message_count = count_user_messages(history)
        if user_message_count >= MAX_USER_MESSAGES:
            print(f"Session {sessionId} reached message limit ({user_message_count}/{MAX_USER_MESSAGES}).")
            # Generate TTS for the predefined message
            try:
                speech_response = client.audio.speech.create(
                    model="tts-1",
                    voice="nova",
                    input=SESSION_LIMIT_MESSAGE
                )
                # Return the audio directly
                return StreamingResponse(
                    speech_response.iter_bytes(), 
                    media_type="audio/mpeg",
                    headers={"X-Session-Id": sessionId, "X-Limit-Reached": "true"} 
                )
            except Exception as tts_error:
                 print(f"Error generating TTS for session limit message: {tts_error}")
                 raise HTTPException(status_code=500, detail="Failed to generate limit notification audio.")
        # --- End Check Message Limit ---

        # Limit not reached, proceed with transcription and AI call
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        audio_file = open(tmp_path, "rb")
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        audio_file.close() # Close file handle explicitly
        transcript = str(transcript_response) 

        print(f"SESSION {sessionId} TRANSCRIPT:", transcript)

        if not transcript or not transcript.strip():
            os.unlink(tmp_path) # Clean up temp file
            raise HTTPException(status_code=400, detail="Empty transcription. Check audio clarity.")

        history.append({"role": "user", "content": transcript})

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history
        )
        reply_text = completion.choices[0].message.content
        print(f"SESSION {sessionId} GPT RESPONSE:", reply_text)

        if not reply_text or not reply_text.strip():
            os.unlink(tmp_path) # Clean up temp file
            if history and history[-1]["role"] == "user":
                 history.pop() # Rollback user message
            raise HTTPException(status_code=400, detail="Empty GPT response. Check prompt and context.")

        history.append({"role": "assistant", "content": reply_text})
        session_histories[sessionId] = history

        speech_response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=reply_text
        )
        
        os.unlink(tmp_path) # Clean up temp file *after* successful processing

        return StreamingResponse(
            speech_response.iter_bytes(), 
            media_type="audio/mpeg",
            headers={"X-Session-Id": sessionId, "X-Limit-Reached": "false"}
        )

    except Exception as e:
        print(f"ERROR in session {sessionId}:", str(e))
        if tmp_path and os.path.exists(tmp_path):
            # Ensure file is closed before unlinking, especially on Windows
            if 'audio_file' in locals() and not audio_file.closed:
                audio_file.close()
            os.unlink(tmp_path)
        # Rollback user message on error if it was added
        if history and history[-1].get("role") == "user" and 'transcript' in locals() and history[-1].get("content") == transcript:
             history.pop()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
         # Ensure file handle is closed if transcription failed before closing
        if 'audio_file' in locals() and not audio_file.closed:
            audio_file.close()


@app.get("/health", summary="Check API health", description="Returns a simple message to check if the API is running.")
async def health_check():
    return {"status": "ok"}

@app.delete("/session/{sessionId}", summary="Clear session history", description="Deletes the conversation history for a given session ID.")
async def clear_session(sessionId: str):
    if sessionId in session_histories:
        del session_histories[sessionId]
        return {"message": f"Session {sessionId} cleared."}
    else:
        raise HTTPException(status_code=404, detail=f"Session {sessionId} not found.")