from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import Translator

app = FastAPI(
    title="Transformer Translation Engine API",
    description="Custom PyTorch Seq2Seq Transformer API for English-to-Italian translation.",
    version="1.0.0"
)

translator = Translator(weights_path="weights/tmodel_24.pt")

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    source_text: str
    translated_text: str

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/translate", response_model=TranslationResponse)
def translate(request: TranslationRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")
    
    try:
        output = translator.translate(request.text)
        return TranslationResponse(
            source_text=request.text,
            translated_text=output
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))