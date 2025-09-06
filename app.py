from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Corpus ampliado con asesoría de marcas y usos
context = """
Componentes de hardware y recomendaciones:

CPU:
- Intel Core i5/i7/i9 son buenos para productividad y gaming.
- AMD Ryzen 5/7/9 ofrecen excelente rendimiento/precio, especialmente en multitarea.
- Para gaming exigente: AMD Ryzen 7 7800X3D o Intel i7-13700K.
- Para oficina: Intel i5 de 12ª/13ª generación.

GPU:
- NVIDIA GeForce RTX 4000 son líderes en gaming y edición de video.
- AMD Radeon RX 7000 ofrecen gran relación calidad/precio.
- Para diseño y edición: NVIDIA RTX 4070/4080.
- Para gaming casual: AMD RX 6600 o NVIDIA RTX 3060.

RAM:
- Corsair, Kingston, G.Skill y Crucial son marcas confiables.
- 16GB es estándar para gaming y multitarea.
- 32GB+ recomendado para edición de video o renderizado 3D.

Almacenamiento:
- SSD NVMe (Samsung 980 Pro, WD Black, Crucial P5 Plus) = máximo rendimiento.
- HDD se usa solo para almacenamiento masivo y barato.

Placa base:
- ASUS, MSI y Gigabyte son marcas recomendadas.
- Compatibilidad con CPU y RAM es clave.

Fuente de poder:
- EVGA, Corsair y Seasonic = alta calidad.
- 650W suele ser suficiente para PCs de gama media.

Periféricos:
- Monitores: ASUS, LG, Dell.
- Teclados/mouses: Logitech, Razer, Corsair.
"""

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]

    keywords = ["cpu","gpu","ram","memoria","procesador","disco","ssd","hdd","fuente","placa","tarjeta","hardware","marca","gaming","oficina","edición"]
    if not any(k in user_message.lower() for k in keywords):
        response = "Solo asesoro sobre hardware de computadoras (CPU, GPU, RAM, SSD, marcas, etc.)."
    else:
        try:
            result = qa_pipeline(question=user_message, context=context)
            response = result["answer"]
        except Exception as e:
            response = f"Ocurrió un error: {str(e)}"

    return render_template("chat.html", user_message=user_message, bot_response=response)

@app.route("/healthz")
def healthz():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

