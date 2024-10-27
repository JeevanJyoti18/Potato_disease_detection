# endpoint => path

from fastapi import FastAPI, File, UploadFile,  HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer

app = FastAPI()

# CORS (Cross-Origin Resource Sharing)
# protocol + domain +port
#  When you use * as a value for the Access-Control-Allow-Origin header, 

# Origins List: Specifies which frontend URLs are allowed to access your backend.
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow_origins=origins: Specifies the list of allowed origins that can access your backend server. In this case, it's allowing http://localhost and http://localhost:3000.
    allow_credentials=True,  #allow_credentials=True: Allows the inclusion of credentials (like cookies, authorization headers, etc.) in the requests from the allowed origins
    allow_methods=["*"],     # allow_methods=["*"]: Allows all HTTP methods (GET, POST, PUT, DELETE, etc.) to be used from the allowed origins. The * wildcard means all methods are allowed.
    allow_headers=["*"],     # allow_headers=["*"]: Allows all headers to be sent from the allowed origins. Headers contain additional information sent with the request, such as Content-Type or Authorization
)

#  'serving_default': This is the default name for the signature that is used for inference
# The serving_default is the most common signature used for inference (making predictions).
MODEL = tf.keras.models.load_model("../saved_models/3.keras" )

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:          # (->) in the function indicate the return type of the function. 
    image = np.array(Image.open(BytesIO(data)))      # Image.open(BytesIO(data)): Opens the binary data as an image.
    return image

@app.post("/predict")                                 # Defines an endpoint for HTTP POST requests at the /predict URL.
async def predict(file: UploadFile = File(...)):      # File(...) is a FastAPI utility that indicates that this endpoint expects a file upload. When a user uploads an image, it will be available in the file parameter.
    image = read_file_as_image(await file.read())     # read_file_as_image(): Converts the binary data to an image.    
    img_batch = np.expand_dims(image, 0)              # np.expand_dims(image, 0), the 0 means "add a new axis at the beginning of the shape"
    
    predictions = MODEL.predict(img_batch)             # Uses the pre-trained model to predict the class of the image.

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # prediction->2D array with 1 row -> [[.1,.7,.2]]
    confidence = np.max(predictions[0])                        # prediction[0] ->1st row -> [.1,.7,.2]
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)



# '''
# Frontend and Backend:

# You have a frontend (the part users interact with) running at http://localhost:8080.
# You have a backend (the server) running at http://localhost (default port 80).
# Cross-Origin Request:

# The frontend wants to talk to the backend.
# Because they run on different ports, the browser sees this as a "cross-origin" request.
# Preflight Request:

# Before the frontend can talk to the backend, the browser sends an HTTP OPTIONS request to the backend. This is called a "preflight" request.
# The browser asks, "Is it okay for me to talk to this backend?"
# Backend Response:

# The backend must respond with specific headers saying, "Yes, it's okay."
# These headers include Access-Control-Allow-Origin which should list http://localhost:8080 to allow the frontend to communicate.
# Allowed Origins List:

# The backend keeps a list of allowed origins (like http://localhost:8080).
# If the frontend's origin is in this list, the backend allows the communication.
# Final Request:

# If the preflight check is successful, the browser lets the frontend send its actual request to the backend.
# '''