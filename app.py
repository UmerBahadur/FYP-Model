from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline

# Path to your model directory
MODEL_PATH = "./model"   # put your model folder inside the repo as /model

app = FastAPI()

# Load model & tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer
)

class EmailRequest(BaseModel):
    text: str
    max_length: int = 80
    min_length: int = 20

@app.post("/summarize")
def summarize_email(req: EmailRequest):
    summary = summarizer(
        req.text,
        max_length=req.max_length,
        min_length=req.min_length,
        do_sample=False
    )
    return {"summary": summary[0]["summary_text"]}

@app.get("/")
def home():
    return {"status": "BART summarizer is running"}
