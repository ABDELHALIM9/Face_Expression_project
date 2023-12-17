import io
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,InputLayer, Activation, Dropout, BatchNormalization

app = FastAPI(title='Deploying a ML Model with FastAPI')
model = tf.keras.models.load_model(r"C:\Users\eid\Downloads\Github-repo\Face_Expression_project\models\model_optimal.h5",compile=False)
class_name = ['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict") 
async def prediction(file: UploadFile = File(...)):

    
    contents = await file.read()
    image = read_file_as_image(contents)
    

    batch = np.expand_dims(image,0)

    predections = model.predict(batch)

    predected_class = class_name[np.argmax(predections[0])]

    img_pil = Image.fromarray(image)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg", headers={"predicted_class": predected_class})


nest_asyncio.apply()


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
# Host depends on the setup you selected (docker or virtual env)
