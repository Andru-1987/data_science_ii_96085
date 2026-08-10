import os
from dotenv import load_dotenv

import json
import pandas as pd
from robyn import Robyn, serve_file, Response
from pyngrok import ngrok

load_dotenv()

# 1. Inicializamos la aplicación
app = Robyn(__file__)

@app.get("/")
def index(request):
    return {
        "api_name": "API de Machine Learning con Robyn",
        "version": "1.0",
        "description": "API de ultra baja latencia para inferencia de ML y descarga de datasets.",
        "status": "online",
        "endpoints": {
            "GET /": "Muestra la información y documentación de la API",
            "POST /predict": "Realiza una predicción. Requiere un JSON con la clave 'features'",
            "GET /download-csv": "Descarga el dataset de entrenamiento/prueba en formato CSV"
        }
    }

# 2. Simulamos tu modelo de Machine Learning
class MiModeloML:
    def __init__(self):
        print("Modelo cargado en memoria (una sola vez).")

    def predecir(self, features):
        return {
            "prediccion": "Aprobado",
            "confianza": 0.94,
            "features_recibidas": features
        }

modelo = MiModeloML()

# 3. Endpoint de Predicción
@app.post("/predict")
def predict(request):
    try:
        cuerpo = request.body
        if isinstance(cuerpo, bytes):
            cuerpo = cuerpo.decode("utf-8")

        datos_entrada = json.loads(cuerpo)
        features = datos_entrada.get("features")

        if not features:
            return Response(
                status_code=400,
                headers={"Content-Type": "application/json"},
                description=json.dumps({"error": "Falta la clave 'features'"})
            )

        resultado = modelo.predecir(features)

        return {
            "estado": "exito",
            "datos": resultado
        }
    except Exception as e:
        return Response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": str(e)})
        )


# 5. Endpoint para servir el CSV
@app.get("/download-csv")
def download_csv(request):
    # 4. Preparación del CSV
    df = pd.DataFrame({
        "id_cliente": [101, 102, 103],
        "edad": [28, 45, 32],
        "score_riesgo": [0.12, 0.85, 0.44]
    })
    df.to_csv("dataset_ml.csv", index=False)
    print("Archivo CSV de prueba generado.")
    return serve_file("dataset_ml.csv")


# --- CONFIGURACIÓN DE NGROK Y ARRANQUE ---

# Pon tu token de ngrok aquí
ngrok.set_auth_token(os.getenv("NGROK_API_KEY"))

# Matamos cualquier túnel previo que haya quedado abierto por error
ngrok.kill()

# Abrimos el túnel en el puerto 8080 (el mismo que usará Robyn)
public_url = ngrok.connect(8080)
print(f"🚀 TU API ESTÁ PÚBLICA EN: {public_url.public_url}")

# Arrancamos Robyn. Esto bloqueará la celda mientras el servidor esté vivo.
app.start(port=8080, host="0.0.0.0")
