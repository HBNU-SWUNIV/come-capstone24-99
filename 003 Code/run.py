from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras_preprocessing.image import img_to_array, load_img
import numpy as np
from tensorflow import keras

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 모델 로드
model = load_model('trained_model_Unet_1.h5')

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(file_path):
    # 이미지 전처리
    image = load_img(file_path, target_size=(256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # 이미지 정규화

    # 예측
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return 'Normal' if predicted_class == 0 else 'Abnormal'

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return redirect(url_for('uploaded_file', filename=filename, prediction=prediction))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>/<prediction>')
def uploaded_file(filename, prediction):
    return f'''
    <!doctype html>
    <title>Prediction Result</title>
    <h1>Prediction Result</h1>
    <p>Filename: {filename}</p>
    <p>Prediction: {prediction}</p>
    <img src="{url_for('send_file', filename=filename)}" alt="Uploaded Image">
    '''

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)