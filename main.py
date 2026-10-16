"""
Class Activity 1 API + CIFAR-10 CNN classifier (Assignment 2).
Run:  uv run fastapi dev    then open http://127.0.0.1:8000/docs
"""
import io
from typing import Union

import spacy
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.bigram_model import BigramModel
from cnn_model import SimpleCNN, CLASSES, IMAGE_SIZE   # NEW

app = FastAPI()

# ======================= Class Activity 1 =======================
# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

nlp = spacy.load("en_core_web_lg")

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vector = nlp(request.word).vector
    return {"word": request.word, "embedding": vector.tolist()}

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    score = nlp(request.word1).similarity(nlp(request.word2))
    return {"word1": request.word1, "word2": request.word2, "similarity": float(score)}


# ======================= NEW: CNN classifier =======================
cnn_model = SimpleCNN()
cnn_model.load_state_dict(torch.load("cnn_cifar10.pth", map_location="cpu"))
cnn_model.eval()  # inference mode

# Must match the preprocessing used in train_cnn.py
cnn_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = cnn_transform(image).unsqueeze(0)          # 1 x 3 x 64 x 64
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {"predicted_class": CLASSES[idx], "confidence": round(float(probs[idx]), 4)}
