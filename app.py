import os
from os import path, walk
from flask import Flask,render_template,redirect, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
# import keras
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

IMG_HEIGHT = 32
IMG_WIDTH = 32
channels = 3
UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

model = tf.keras.models.load_model("./static/model/GTSRB.h5")

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
app = Flask(__name__,template_folder="./templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app_data = {
    "name":         "German Traffic Sign Recogition Benchmark ( GTSRB )",
    "description":  "GTSRB Dataset trained on Convolutional Neural Network ",
    "author":       "Jayanth Kumar",
    "html_title":   "German Traffic Sign Recogition Benchmark ( GTSRB )",
    "project_name": "Traffic Sign Classifier",
    "keywords":     "GTSRB, CNN, ANN"
}
@app.route('/')
def home():
    return render_template('index.html', app_data=app_data)
    
@app.route('/predict/', methods = ['GET', 'POST'])
def upload_image():
    # if len(request.files) ==0:
    #     return render_template('home.html', app_data=app_data)
    # print(request.files)
    # file = request.files['file']
    # print()
    # if file.filename == '':
    #     print('No image selected for uploading')
    #     return render_template('home.html', app_data=app_data)
    # if file and allowed_file(file.filename):
    #     filename = secure_filename(file.filename)
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     print('Image successfully uploaded and displayed below')

    #     pred=test_input(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     print("Predicton ===== > ",pred)

    #     image_path="."+os.path.join(app.config['UPLOAD_FOLDER'], filename)
    #     print(image_path)
    #     return render_template('predict.html', app_data=app_data,image_path=image_path,pred=pred)
    # else:
    #     print('Allowed image types are -> png, jpg, jpeg, gif')
    return render_template('predict.html')

def test_input(path):
    image = cv2.imread(path)
    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
    X_test = np.array(resize_image)
    X_test = X_test/255
    pred=np.argmax(model.predict(X_test.reshape(1,32,32,3)))
    return classes[int(pred)]

extra_dirs = ['./static/styles','./templates']
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)

if __name__ == '__main__':
    app.run(debug=True,extra_files=extra_files)