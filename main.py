from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import carregar_modelo


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = carregar_modelo()
    yield
    del app.state.model


app = FastAPI(
    title="API de Classificação de Dígitos",
    description="API que utiliza uma CNN treinada com o MNIST.",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


@app.get("/")
def inicio():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"mensagem": "API de classificação de dígitos funcionando!"}


@app.post("/predict")
def prever_numero(request: Request, file: UploadFile = File(...)):
    try:
        with Image.open(file.file) as imagem:
            imagem_tensor = transform(imagem.convert("L")).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as erro:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível processar a imagem. Envie uma imagem válida.",
        ) from erro
    finally:
        file.file.close()

    with torch.inference_mode():
        outputs = request.app.state.model(imagem_tensor)
        probabilidades = torch.softmax(outputs, dim=1)
        numero = probabilidades.argmax(dim=1).item()
        confianca = probabilidades[0, numero].item()

    return {
        "numero_previsto": numero,
        "confianca": round(confianca, 4),
    }
