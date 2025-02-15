import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('my_model.h5') #loading a trained model

st.write("""
         # egg tart salmon sashimi Unknown Prediction
         """
         )

st.write("This is a simple image classification web app to predict eggtart-salmonsashimi and unknowns")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

st.write("0 for egg tart, 1 for salmon sashimi, 2 for unknown")
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is a egg tart!")
    elif np.argmax(prediction) == 1:
        st.write("It is a salmon sashimi!")
    else:
        st.write("It is unknown!")

    st.write(prediction)
