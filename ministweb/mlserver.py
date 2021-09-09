from flask import Flask
from flask import render_template
from flask import request
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras import backend as K

app = Flask(__name__ , static_url_path='')

@app.route("/")
def root(name=None):
    return render_template('predict.html', name=name)


@app.route("/predict", methods=['POST', 'GET'])
def predict():

    canvas = request.form['images']
    print( canvas)
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.shape)
    print(train_labels.shape)

    test_images=test_images.reshape((10000,28,28,1))
    test_images=test_images.astype('float32')/255
    test_labels[0]
    test_labels=to_categorical(test_labels)

    print(test_labels[0])
    print(test_labels.shape)

    new_model=tf.keras.models.load_model('./mnistCNN.h5')
    new_model.summary()
    new_model.evaluate(x= test_images, y=test_labels)


    lcanvas = canvas.split()
    lcanvas = [float(v) for v in lcanvas]


    predict=new_model.predict(np.array(lcanvas).reshape(1,28,28,1))
    print(predict)
    K.clear_session()
    p_val=np.argmax(predict)
    print(p_val)
    # 이미 그래프가 있을 경우 중복이 될 수 있기 때문에, 기존 그래프를 모두 리셋한다.
    #tf.reset_default_graph()

    
    print('\nreload has been done\n')
    #fcanvas = [float(z) for z in canvas.split(",")]

  

    #fcanvas = [float(list_item) for list_item in lcanvas]
    #fcanvas = []

    #for i in range(len(lcanvas)-1):
    #   fcanvas[i] = float(lcanvas[i])

    print(lcanvas)

   # p_val = sess.run(p, feed_dict={x: [lcanvas], keep_prob: 1.0})
   # print("p_val: ", p_val)

   

    return render_template('predict.html', result=p_val)


if __name__ == "__main__":
    app.run(host='0.0.0.0')