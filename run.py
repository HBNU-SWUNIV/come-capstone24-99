from flask import Flask, request, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# 저장된 모델 로드 (binary classification model and 4-class classification model)
binary_model = load_model('trained_model_ResNet50.h5')
multi_class_model = load_model('trained_model_ResNet50_4class.h5')

# 클래스 이름 (정상, 비정상)
binary_class_names = ['비정상', '정상']

# 클래스 이름 (Normal, Adenocarcinoma, Large_cell_carcinoma, Squamous_cell_carcinoma for 4-class model)
multi_class_names = ['선암', '대세포암', '정상', '편평세포암']


# 이미지 전처리 함수 (모델과 동일한 전처리 방식)
def preprocess_image(image_path, target_size=(224, 224)):
    # 이미지 로드
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)

    # 모델 입력에 맞게 이미지 스케일링
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    image = image / 255.0  # 0~1 범위로 스케일링
    return image


# Flask 루트 경로에서 HTML 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 이미지 업로드 및 분류
@app.route('/predict_binary', methods=['POST'])
def predict_binary():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        # 업로드된 이미지 저장
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # 이미지 전처리
        image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # 예측
        predictions = binary_model.predict(image)
        predicted_class = 1 if predictions > 0.5 else 0  # 0.5 기준으로 클래스 결정

        # 클래스 이름 반환
        predicted_label = binary_class_names[predicted_class]

        # 결과를 HTML 페이지로 반환
        return render_template('result.html', label=predicted_label, file_path=filename)

# 이미지 업로드 및 4-class 분류 (Normal, Adenocarcinoma, Large_cell_carcinoma, Squamous_cell_carcinoma)
@app.route('/predict_multi', methods=['POST'])
def predict_multi():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        # 업로드된 이미지 저장
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # 이미지 전처리
        image = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # 예측 (multi-class classification)
        predictions = multi_class_model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]  # 예측된 클래스의 인덱스를 반환

        # 클래스 이름 반환
        predicted_label = multi_class_names[predicted_class]

        # 결과를 HTML 페이지로 반환
        return render_template('result_multi.html', label=predicted_label, file_path=filename)



# 업로드된 파일을 정적으로 서빙하는 라우트 (추가)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
