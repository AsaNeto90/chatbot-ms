from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from openai import OpenAI
import os

app = FastAPI(title="Chatbot API", description="An API to access an OpenAI chat.", version="0.1.0")

# Set your OpenAI API key as an environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Question(BaseModel):
    context: str
    text: str

class Answer(BaseModel):
    response: str

@app.post("/ask", response_model=Answer, summary="Ask the AI a question. Pass in the context you want the API to base itself on.", description="Sends a text question to OpenAI and returns the generated response.")
async def ask_holocron(question: Question):
    try:
        response = completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": question.context},
                {"role": "user", "content": question.text}
            ]
        )
        ai_response = completion.choices[0].message.content
        return {"response": ai_response}
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=str(e)) # Handle OpenAI API errors
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.") # Generic error handling

@app.get("/health", summary="Check API health", description="Returns a simple message to check if the API is running.")
async def health_check():
    return {"status": "ok"}