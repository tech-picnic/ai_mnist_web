# 관련 라이브러리 불러옴
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

#root에서 predict.html 파일 랜더링
@app.route("/")
def root(name=None):
    return render_template('predict.html', name=name)


@app.route("/predict", methods=['POST', 'GET'])

#예측값 함수 정의
def predict():
#폼의 이미지를 canvas에 대입
    canvas = request.form['images']
    print( canvas)
    #mnist 데이터를 로딩함
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.shape)
    print(train_labels.shape)
# 이미지 데이터 전처리
    test_images=test_images.reshape((10000,28,28,1))
    test_images=test_images.astype('float32')/255
    test_labels[0]
    test_labels=to_categorical(test_labels)

    print(test_labels[0])
    print(test_labels.shape)
# 미리 만들어 놓은 가중치 파일을 로딩
    new_model=tf.keras.models.load_model('./mnistCNN.h5')
    new_model.summary()
    new_model.evaluate(x= test_images, y=test_labels)

#캔버스값들을 소수형으로 변환
    lcanvas = canvas.split()
    lcanvas = [float(v) for v in lcanvas]

#리쉐입해서 new_model에 집어넣어 predict하고(정확한 값이 가장 높은 수가 나옴) np.argmax로 가장 높은 숫자를 예측값으로  p_val로 저장
    predict=new_model.predict(np.array(lcanvas).reshape(1,28,28,1))
    print(predict)
    K.clear_session()
    p_val=np.argmax(predict)
    print(p_val)
    # 이미 그래프가 있을 경우 중복이 될 수 있기 때문에, 기존 그래프를 모두 리셋한다.
    #tf.reset_default_graph()    
    print('\nreload has been done\n')
    print(lcanvas)
   # p_val은 예측한 결과값을 predict.html에 p_val변수로 출력 
    return render_template('predict.html', result=p_val)


if __name__ == "__main__":
    app.run(host='0.0.0.0')