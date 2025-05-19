from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import src.retrieve_llm_with_memory as chatbot_module  # Adjust path if needed

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# For simplicity, using a hardcoded session ID (can be dynamic later)
SESSION_ID = "user-1"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": "", "user_input": ""})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    try:
        result = chatbot_module.chat_with_memory.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": SESSION_ID}},
        )
        response = result.content
    except Exception as e:
        response = f"Error: {str(e)}"
    return templates.TemplateResponse("index.html", {"request": request, "response": response, "user_input": user_input})
