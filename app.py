from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import cv2
import numpy as np
import os
from datetime import datetime
import face_recognition
import pickle
import requests
os.system("pip install dlib-bin")

app = FastAPI()

# === CONFIG ===
FIREBASE_PKL_URL = "https://firebasestorage.googleapis.com/v0/b/testcitrufy.appspot.com/o/face_encodings.pkl?alt=media"
LOCAL_PKL_PATH = "face_encodings.pkl"

# === Load known encodings from Firebase or local cache ===
def load_encodings():
    try:
        if not os.path.exists(LOCAL_PKL_PATH):
            print("⬇️ Downloading face_encodings.pkl from Firebase Storage...")
            r = requests.get(FIREBASE_PKL_URL)
            if r.status_code == 200:
                with open(LOCAL_PKL_PATH, "wb") as f:
                    f.write(r.content)
                print("✅ Successfully downloaded encodings file.")
            else:
                raise Exception(f"Failed to download file (HTTP {r.status_code})")

        print("✅ Loading encodings from local file...")
        with open(LOCAL_PKL_PATH, "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]

    except Exception as e:
        print(f"❌ Error loading encodings: {e}")
        return [], []

# Load encodings at startup
known_encodings, known_names = load_encodings()

@app.get("/", response_class=HTMLResponse)
def upload_form():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Face Image Collector</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f8fafc;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 0 15px rgba(0,0,0,0.1);
                width: 360px;
                text-align: center;
            }
            input[type="text"], input[type="file"] {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
            button {
                background-color: #2563eb;
                color: white;
                border: none;
                padding: 10px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                width: 100%;
            }
            button:hover {
                background-color: #1d4ed8;
            }
            .message {
                margin-top: 15px;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Upload Person Image</h2>
            <input type="text" id="name" placeholder="Enter person name" required />
            <input type="file" id="file" accept="image/*" required />
            <button onclick="uploadImage()">Upload</button>
            <div class="message" id="message"></div>
        </div>

        <script>
            async function uploadImage() {
                const name = document.getElementById("name").value.trim();
                const fileInput = document.getElementById("file");
                const messageEl = document.getElementById("message");

                if (!name || !fileInput.files.length) {
                    messageEl.style.color = "red";
                    messageEl.textContent = "Please enter a name and select an image.";
                    return;
                }

                const formData = new FormData();
                formData.append("name", name);
                formData.append("file", fileInput.files[0]);

                messageEl.style.color = "black";
                messageEl.textContent = "Uploading...";

                try {
                    const response = await fetch("/collect", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();

                    if (response.ok) {
                        messageEl.style.color = "green";
                        messageEl.textContent = "✅ Upload successful! Saved at: " + data.path;
                        fileInput.value = "";
                        document.getElementById("name").value = "";
                    } else {
                        messageEl.style.color = "red";
                        messageEl.textContent = "❌ Error: " + (data.error || "Upload failed");
                    }
                } catch (err) {
                    messageEl.style.color = "red";
                    messageEl.textContent = "❌ Network error: " + err.message;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        result = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            if True in matches:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match = np.argmin(face_distances)
                name = known_names[best_match]
            result.append(name)

        return {"recognized": result}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/collect")
async def collect_person_image(name: str = Form(...), file: UploadFile = File(...)):
    try:
        person_dir = os.path.join("dataset", name)
        os.makedirs(person_dir, exist_ok=True)

        img_bytes = await file.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image format"})

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(person_dir, filename)
        cv2.imwrite(filepath, image)

        return {"status": "success", "path": filepath}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    # For Render: bind to 0.0.0.0 and use PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 
