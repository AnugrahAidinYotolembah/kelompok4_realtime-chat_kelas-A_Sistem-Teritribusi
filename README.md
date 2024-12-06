# AZTEC APPS :  Real-time Chat Hate Speech Detection 

### *Informasi Proyek*
- *Kelompok*: 4  
- *Topik*: Realtime Chat Hate Speech Detection  
- *Kelas*: A  
- *Mata Kuliah*: Sistem Terdistribusi  

### *Anggota Kelompok*
1. *Anugrah Aidin Yotolembah*  
2. *Zaky Nur Abyan*

### *A. UI Interface*

#### *1. Login*
<img width="1512" alt="Screenshot 2024-12-06 at 09 14 33" src="https://github.com/user-attachments/assets/cbbedb20-b1c2-4acc-bdb6-2d6162cc8431">

#### *2. room chat*
<img width="1512" alt="Screenshot 2024-12-06 at 09 21 48" src="https://github.com/user-attachments/assets/552d50ee-eccc-423a-9231-f638ca1449ac">

#### *3. Hate speech detection in chat*
<img width="1512" alt="Screenshot 2024-12-06 at 09 22 29" src="https://github.com/user-attachments/assets/1499b804-782b-4c45-9d30-df47b9776e8e">
<img width="1512" alt="Screenshot 2024-12-06 at 09 22 40" src="https://github.com/user-attachments/assets/7e20b610-caae-4c5a-8b37-c4c17552e2ac">

#### *4. logout*
<img width="1512" alt="Screenshot 2024-12-06 at 09 14 33" src="https://github.com/user-attachments/assets/cbbedb20-b1c2-4acc-bdb6-2d6162cc8431">

### *B. Main Program*
'''
import asyncio
import csv
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import paho.mqtt.client as mqtt
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the Hate Speech detection model and tokenizer from Hugging Face
tokenizer = None
model = None
hate_speech_model = None

try:
    tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-indonesian")
    model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-indonesian")
    hate_speech_model = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print("Model dan Tokenizer berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model atau tokenizer: {e}")

# Memuat dataset CSV yang berisi kalimat ujaran kebencian (kolom abuse)
dataset_path = "/Users/ayotolembah/Documents/Kerja/Quadrant/realtime chat/realtime_chat/abusive copy.csv"  # Ganti dengan path file CSV Anda
df = pd.read_csv(dataset_path)

# Fungsi untuk memuat kalimat dan kata-kata kasar dari dataset
def load_abuse_dataset(df):
    abuse_sentences = df['abuse'].dropna().tolist()
    abuse_keywords = df['abuse_keywords'].dropna().tolist()
    return abuse_sentences, abuse_keywords

abuse_sentences, abuse_keywords = load_abuse_dataset(df)

# Fungsi deteksi hate speech dengan memperkaya kosa kata
def detect_hate_speech(text: str):
    if not isinstance(text, str):
        return False
    
    try:
        # Cek apakah kalimat ada dalam daftar abuse_sentences
        if text.lower() in [s.lower() for s in abuse_sentences]:
            print(f"Detected hate speech (from sentence): {text}")
            return True
        
        # Memecah kalimat menjadi kata-kata
        words = text.split()

        # Cek apakah salah satu kata ada dalam daftar abuse_keywords
        for word in words:
            if word.lower() in [keyword.lower() for keyword in abuse_keywords]:
                print(f"Detected hate speech (from keyword): {word}")
                return True
        
        # Jika tidak ada, gunakan model untuk mendeteksi
        result = hate_speech_model(text)
        label = result[0]["label"]
        print(f"Prediction label: {label}")
        return label == "LABEL_1"  # 'LABEL_1' indicates hate speech
    except Exception as e:
        print(f"Error dalam mendeteksi ujaran kebencian: {e}")
        return False

# Manajemen koneksi WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# MQTT setup
mqtt_client = mqtt.Client()

# MQTT callback when a message is received
def on_message(client, userdata, msg):
    print(f"Received MQTT message: {msg.payload.decode()}")

mqtt_client.on_message = on_message
mqtt_client.connect("test.mosquitto.org", 1883, 60)
mqtt_client.subscribe("chat_topic")

# Start the MQTT client in a separate thread to prevent blocking the FastAPI server
def start_mqtt():
    mqtt_client.loop_start()

@app.on_event("startup")
async def startup():
    threading.Thread(target=start_mqtt, daemon=True).start()

# Serve the Enter Name page
@app.get("/", response_class=HTMLResponse)
async def enter_name():
    with open("/Users/ayotolembah/Documents/Kerja/Quadrant/realtime chat/realtime_chat/login.html", "r") as file:  # Ganti dengan path yang sesuai
        return file.read()

# Serve the Chat Room page after entering name
@app.get("/chat", response_class=HTMLResponse)
async def chat_room():
    with open("/Users/ayotolembah/Documents/Kerja/Quadrant/realtime chat/realtime_chat/chat_room.html", "r") as file:  # Ganti dengan path yang sesuai
        return file.read()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()

            # Deteksi hate speech
            if detect_hate_speech(data):
                # Kirim pesan tentang hate speech terdeteksi ke semua client tanpa nama pengirim
                message = "Hate speech detected. Message blocked."
                await manager.broadcast(message)

                # Simpan pesan yang diblokir ke dalam file CSV dengan timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('hate_speech_history.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, client_id, data, "Hate speech detected"])
                continue  # Skip broadcasting message jika hate speech terdeteksi

            # Lakukan prediksi hate speech dan kirim hasil label ke frontend
            result = hate_speech_model(data)
            label = result[0]["label"]
            print(f"Prediction label: {label}")

            # Kirim label prediksi ke frontend jika bukan non-hate
            if label != "LABEL_0":  # Jika bukan NON_HATE, kirim ke frontend
                await manager.broadcast(f"Prediction label: {label}")
            
            # Kirim pesan biasa ke semua client lain
            await manager.broadcast(f"{client_id}: {data}")  # Menampilkan nama client tanpa "Client #"
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} has left the chat")  # Ganti format client dengan client_id saja
'''

