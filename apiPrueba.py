
##LA IDEA ES OBTENER EL URL DE LA IMAGEN DESDE CLOUDIINARY QUE SUBIO LA PERSONA EN EL FORNT 
##IDEAL: MANDARLE POR API LA URL DIRECTO... Y ANALIZARLA CON EL MODELO
import tensorflow as tf
import cloudinary
import cloudinary.uploader
import cloudinary.api
import logging
import os
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask import jsonify
from flask import Flask,render_template, request
from cloudinary.utils import cloudinary_url
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras



load_dotenv()
##pip install python-dotenv

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
#verify cloud
app.logger.info('%s',os.getenv('dhnzx75kb'))

import cloudinary
import requests

cloudinary.config( 
  cloud_name = "dhnzx75kb", 
  api_key = "929195717772384", 
  api_secret = "KZ5r_n5zrMDLa31aS6bZ4PIrJI0" 
)



from keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO

def preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((256, 256))  # Cambia el tamaño según las necesidades de tu modelo
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalización
    return img_array



##https://res.cloudinary.com/dhnzx75kb/image/upload/v1692562632/P_REAL_0013_nih1bo.png
@app.route('/get_image', methods=['GET'])
def get_image():
        image_url = request.args.get('image_url')
        # Transforma la imagen al formato JPG usando Cloudinary
        ##jpg_image_url = cloudinary.CloudinaryImage(image_url).image(format='jpg')

        
        img_array = preprocess_image(image_url)
        model = load_model('modelo_tipo_manzanas.h5')
        prediction = model.predict(img_array)


        ##preprocessed_image = preprocess_image(jpg_image_url)
       ## prediction = model.predict(preprocessed_image)
        return jsonify({'prediction': prediction.tolist()})
       
       



# home route
@app.route('/')
def hello():
    return "Hello World!"