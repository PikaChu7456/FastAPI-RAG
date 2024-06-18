from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cohere

# Initialize Cohere client
co = cohere.Client('TcZjPcNuntkBpDbSsH5M5X8N9vlSs6Mq11KoL3rd')

# Initialize FastAPI app
app = FastAPI()

# In-memory chat history
chat_history = []

# Request and Response Models
class Message(BaseModel):
    text: str

class Response(BaseModel):
    reply: str

@app.post("/chat", response_model=Response)
def chat(message: Message):
    try:
        # Generate a response with the current chat history
        response = co.chat(
            model='command-r-plus',
            prompt_truncation='AUTO',
            connectors=[],
            message=message.text,
            temperature=0.8,
            chat_history=chat_history,
            preamble='Humorous, witty, and playful. Think comedy writer.'
        )
        answer = response.text

        # Add message and answer to the chat history
        user_message = {"role": "USER", "text": message.text}
        bot_message = {"role": "CHATBOT", "text": answer}

        chat_history.append(user_message)
        chat_history.append(bot_message)

        return Response(reply=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
