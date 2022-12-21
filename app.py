import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dropout, Conv2D, Rescaling
from keras.layers import Dense


app = Flask(__name__)

classifier = Sequential()
classifier.add(Convolution2D(filters=64, kernel_size=(3, 3), input_shape=(80, 80, 3), kernel_initializer='he_uniform', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(5, 5)))
classifier.add(Flatten())
classifier.add(Dense(units=62, kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(units=3, kernel_initializer='glorot_uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        prediction = ""
        file = request.files['query_img']
        img = Image.open(file.stream)
        uploaded_img_path = "static/images_cherchees/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)
        test_image = keras.utils.load_img(uploaded_img_path, target_size = (80, 80))
        test_image = keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        
        stTime = datetime.now()
        result = classifier.predict(test_image)
        t = datetime.now() - stTime
        tStr = f"{t}"
        time = f"{tStr[tStr.rindex(':')+1:]} "

        if result[0][0] == 0:
            prediction = 'benign'
        elif result[0][0] == 1:
            prediction = 'malignant'
        else:
            prediction = 'normal'


        return render_template('index.html', query_path = uploaded_img_path, prediction = prediction, time = time)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run()
