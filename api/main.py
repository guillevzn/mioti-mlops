import os
from fastapi import FastAPI, Request, HTTPException,  Depends, status
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from BuyerClassification import BuyerClassification
import sqlite3

database_path = "database/database.db"

# Cargar el modelo
loaded_model = BuyerClassification.load_model('BuyerClassification.joblib')

# Leer la API_KEY del archivo
try:
    with open('.creds/API_KEY', 'r') as f:
        API_KEY = f.read().strip()
except FileNotFoundError:
    raise Exception("API_KEY file not found. Please create .creds/API_KEY file with your API key.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header   
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

app = FastAPI()

# Configurar Jinja2Templates
templates = Jinja2Templates(directory="templates/")

class Event(BaseModel):
    event_time: str
    event_type: str
    product_id: str
    category_id: str
    category_1: str
    brand: str
    price: float
    user_id: int
    user_session: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Obtener las categorías únicas
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT category_1 FROM categories")
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return templates.TemplateResponse("store.html", {"request": request, "categories": categories})

@app.get("/events/{category}")
async def get_events(category: str, api_key: str = Depends(get_api_key)):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Obtener los eventos del usuario aleatorio para la categoría seleccionada
    cursor.execute(f"""
        WITH random_user AS (
            SELECT user_id
            FROM selected_events
            WHERE category_1 = ?
            ORDER BY RANDOM()
            LIMIT 1
        )
        SELECT
            event_time,
            event_type,
            product_id,
            category_id,
            category_1,
            brand,
            price,
            user_id,
            user_session
        FROM selected_events
        WHERE
            user_id = (SELECT user_id FROM random_user)
            AND event_type != 'purchase'
            AND category_1 = ?
        ORDER BY event_time;
    """, (category, category))
    
    events = [Event(
        event_time=row[0],
        event_type=row[1],
        product_id=row[2],
        category_id=row[3],
        category_1=row[4],
        brand=row[5],
        price=row[6],
        user_id=row[7],
        user_session=row[8]
    ).dict() for row in cursor.fetchall()]

    conn.close()
    print(events)
    return {"events": events}

@app.post("/predict")
async def predict(events: list[Event], api_key: str = Depends(get_api_key)):
    try:
        # Convertir la lista de eventos a un DataFrame
        df = pd.DataFrame([event.dict() for event in events])
        
        # Hacer las predicciones
        predictions, probabilities = loaded_model.predict(df)
        
        # Devolver las predicciones y probabilidades
        return {"predictions": predictions.tolist(), "probabilities": probabilities.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)