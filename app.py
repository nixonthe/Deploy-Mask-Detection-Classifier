import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/mask_model.h5'

# Load your trained model
model = tf.keras.models.load_model(MODEL_PATH)
model.make_predict_function()
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f.save('file.jpg')

        # Make prediction
        preds = model_predict('file.jpg', model)

        # Process your result for human
        classes = ['mask_weared_incorrect', 'with_mask', 'without_mask']
        result = classes[np.argmax(preds)]

        return result

    return None


if __name__ == '__main__':
    app.run(debug=True)