import os
from flask import Flask, redirect, url_for, render_template, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    target = app.config['UPLOAD_FOLDER']
    if 'img' not in request.files:
        return "No file part"
        
    file = request.files['img']
    if file.filename == '':
        return "No selected file"
        
    filename = file.filename
    file_path = os.path.join(target, filename)
    file.save(file_path)

    # Convert to JPEG if necessary
    if file_path.endswith('.png'):
        img = Image.open(file_path).convert('RGB')
        file_path_jpg = os.path.splitext(file_path)[0] + '.jpg'
        img.save(file_path_jpg, 'JPEG')
        os.remove(file_path)
        file_path = file_path_jpg
        filename = os.path.basename(file_path_jpg)
    
    print("Upload success")
    return jsonify({'filename': filename})

@app.route('/predic/<filename>', methods=['GET', 'POST'])
def predic(filename):
    gambar = url_for('custom_static', filename=filename)
    print("URL Gambar:", gambar)
    hasil = process_image(filename)
    return render_template('predic.html', hasil=hasil, data=filename)

def process_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print("File Path:", file_path)
    
    if os.path.exists(file_path):
        print("File exists")
        img = Image.open(file_path).resize((224, 224)).convert('RGB')
        print("Image size before preprocessing:", img.size)
        
        img = np.array(img) / 255.0  # Normalisasi gambar
        
        print("Image shape after preprocessing:", img.shape)
        
        # Memastikan gambar memiliki dimensi batch
        img = np.expand_dims(img, axis=0).astype(np.float32)  # Ubah tipe data ke FLOAT32
        
        # Berikan input kepada model
        interpreter.set_tensor(input_details[0]['index'], img)

        # Jalankan inferensi
        interpreter.invoke()

        # Dapatkan hasil prediksi
        prediksi = interpreter.get_tensor(output_details[0]['index'])
        
        # Dekode hasil prediksi
        kelas = ['yeji', 'sana', 'jihyo', 'irene', 'eunha']
        hasil_index = np.argmax(prediksi)
        hasil = kelas[hasil_index]
        
        return hasil
    else:
        print("File does not exist")
        return "File not found"

@app.route('/uploads/<path:filename>')
def custom_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
