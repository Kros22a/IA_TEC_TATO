import os
import threading
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Modelo chico por defecto para que descargue rápido
MODEL_ID = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")

# Variables de caché; evitamos usar '/.cache'
HF_HOME = os.environ.get("HF_HOME", "/app/.cache/huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "transformers"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_HOME, "hub"))

generator = None
_model_lock = threading.Lock()

def get_generator():
    global generator
    if generator is None:
        with _model_lock:
            if generator is None:
                os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
                os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
                generator = pipeline("text-generation", model=MODEL_ID)
    return generator

@app.get("/healthz")
def healthz():
    # Health check MUY rápido: no dispara descarga del modelo
    return {"status": "ok"}, 200

@app.post("/warmup")
def warmup():
    # Descarga y prepara el modelo, útil tras desplegar
    gen = get_generator()
    _ = gen("hola", max_new_tokens=5)
    return {"warmed": True}, 200

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    prompt = ""
    if request.method == "POST":
        prompt = request.form.get("text", "")
        if prompt:
            gen = get_generator()
            out = gen(prompt, max_new_tokens=50, num_return_sequences=1)[0]["generated_text"]
            result = out
    return render_template("index.html", result=result, prompt=prompt)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

