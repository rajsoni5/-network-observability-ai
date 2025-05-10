from fastapi import FastAPI
import subprocess
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request, Form

templates = Jinja2Templates(directory="backend/templates")



app = FastAPI()

# Load model on startup
MODEL_PATH = "backend/anomaly_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

class NetStat(BaseModel):
    bytes_sent_diff: float
    bytes_recv_diff: float
    
class PredictionRequest(BaseModel):
    byte_count_diff: int
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "result": None})

@app.post("/dashboard", response_class=HTMLResponse)
def post_dashboard(request: Request,
                   bytes_sent_diff: float = Form(...),
                   bytes_recv_diff: float = Form(...)):
    if model is None:
        result = None
    else:
        X = np.array([[bytes_sent_diff, bytes_recv_diff]])
        prediction = model.predict(X)[0]
        result = prediction == -1
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "result": result
    })

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Your anomaly detection logic here
    return {"is_anomaly": False, "prediction": 1}

@app.post("/predict")
def predict_anomaly(stat: NetStat):
    if model is None:
        return {"error": "Model not loaded."}
    
    X = np.array([[stat.bytes_sent_diff, stat.bytes_recv_diff]])
    prediction = model.predict(X)[0]  # -1 for anomaly, 1 for normal
    return {
        "is_anomaly": prediction == -1,
        "prediction": int(prediction)
    }
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Network Observability Project!"}

@app.get("/ping/{host}")
def ping(host: str):
    try:
        result = subprocess.run(['ping', '-n', '4', host], capture_output=True, text=True)
        return {"host": host, "ping_output": result.stdout}
    except Exception as e:
        return {"error": str(e)}
import psutil

@app.get("/netstats")
def get_net_stats():
    net_io = psutil.net_io_counters()
    return {
        "bytes_sent": net_io.bytes_sent,
        "bytes_recv": net_io.bytes_recv,
        "packets_sent": net_io.packets_sent,
        "packets_recv": net_io.packets_recv
    }
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    byte_count_diff: int

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Your anomaly detection logic here
    return {"is_anomaly": False, "prediction": 1}

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "result": None})

@app.post("/dashboard", response_class=HTMLResponse)
def post_dashboard(request: Request,
                   bytes_sent_diff: float = Form(...),
                   bytes_recv_diff: float = Form(...)):
    if model is None:
        result = None
    else:
        X = np.array([[bytes_sent_diff, bytes_recv_diff]])
        prediction = model.predict(X)[0]
        result = prediction == -1
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "result": result
    })
@app.get("/live-stats")
def live_stats():
    import psutil

    net = psutil.net_io_counters()
    bytes_sent = net.bytes_sent
    bytes_recv = net.bytes_recv
    packets_sent = net.packets_sent
    packets_recv = net.packets_recv

    # You could also integrate the model prediction here
    return {
        "bytes_sent": bytes_sent,
        "bytes_recv": bytes_recv,
        "packets_sent": packets_sent,
        "packets_recv": packets_recv
    }
@app.get("/live-stats")
def live_stats():
    import psutil

    net = psutil.net_io_counters()
    bytes_sent = net.bytes_sent
    bytes_recv = net.bytes_recv

    # Use previous values to compute diff (you may want to cache this properly)
    # For simplicity, use static previous values (or implement global cache)
    if not hasattr(app.state, "prev_bytes"):
        app.state.prev_bytes = (bytes_sent, bytes_recv)
        return {}

    prev_sent, prev_recv = app.state.prev_bytes
    sent_diff = bytes_sent - prev_sent
    recv_diff = bytes_recv - prev_recv
    app.state.prev_bytes = (bytes_sent, bytes_recv)

    # Predict anomaly
    is_anomaly = False
    if model:
        import numpy as np
        prediction = model.predict(np.array([[sent_diff, recv_diff]]))[0]
        is_anomaly = prediction == -1

    return {
        "bytes_sent": bytes_sent,
        "bytes_recv": bytes_recv,
        "sent_diff": sent_diff,
        "recv_diff": recv_diff,
        "anomaly": is_anomaly
    }


