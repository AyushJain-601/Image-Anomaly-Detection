import os
from tqdm import tqdm
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split


Orig = "./dataset/thumb.jpeg"
posDIR = "./dataset/Positive/"
testDIR = "./dataset/Testdataset/"
imgSize = 128
lr = 0.001

MODEL_NAME = 'Image-Forgery-{}-{}.model'.format(lr, 'conv-basic')

#Labeling the data so that output can be classified based on their for i.e. positive or negative
def data_label(imgtag):
    address = imgtag
    word_label = address.split(".")[0]
    if word_label == "positive":
        return np.array([1,0])

    elif word_label == "negative":
        return np.array([0,1])


#Function to convert images to their ELA form so that the CNN can be trained on these ELA images
def convert_ela(path, qual):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=qual)
    temp = Image.open(resaved_filename)
    diff = ImageChops.difference(im, temp)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])

    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    diff = ImageEnhance.Brightness(diff).enhance(scale)
    return diff



#Global training, testing dataset and label
x = []
y = []
x_train = []
y_train = []
x_val = []
y_val = []

def training_data(direc):
    trainingData = []

    for img in tqdm(os.listdir(direc)):
        label = data_label(img)   #Labeling training images based on their types positive or negative
        pathImg = os.path.join(direc, img)
        elaImg = convert_ela(pathImg, 90)

        x.append(np.array(elaImg.resize((128, 128))).flatten() / 255.0)
        y.append(label)
    trainingData.append([x,y])
    # print(trainingData)
    np.save('training_data.npy', trainingData)
    return



x_test = []
y_test = []
def testing_data(direc):
    testingData= []

    for img in tqdm(os.listdir(direc)):
        pathImg = os.path.join(direc, img)
        elaImg = convert_ela(pathImg, 90)
        num = pathImg.split('.')[2]
        print(num)
        x_test.append(np.array(elaImg.resize((128, 128))).flatten() / 255.0)
        y_test.append(num)
    testingData.append([x_test,y_test])
    np.save('testing_data.npy', testingData)
    return testingData



if __name__ == '__main__':
    traindata = training_data(posDIR)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=3)

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_val = x_val.reshape(-1, imgSize, imgSize, 3)
    x_train = x_train.reshape(-1, imgSize, imgSize, 3)

    print(np.array(x_val).shape)
    print(np.array(x_train).shape)

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    import tensorflow as tf

    tf.reset_default_graph()
    convnet = input_data(shape=[None, imgSize, imgSize, 3], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=lr,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_verbose=3)

    history = model.fit({'input': x_train},{'targets': y_train}, n_epoch=30,
              validation_set=({'input':x_val},{'targets': y_val}), snapshot_step=500,
              show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)


    for img in tqdm(os.listdir(testDIR)):
        pathImg = os.path.join(testDIR, img)
        elaImg = convert_ela(pathImg, 90)
        num = pathImg.split('.')[2]
        x_test.append(np.array(elaImg.resize((128, 128))).flatten() / 255.0)
        y_test.append(num)

    x_test = np.array(x_test)
    x_test = x_test.reshape(-1, imgSize, imgSize, 3)
    print(y_test)
    output = []
    i=0;
    for img in x_test:
        img = np.expand_dims(img, 0)
        pred = model.predict(img)
        pred2 = (model.predict(img) > 0.5).astype(int)
        if pred2[0][0] == 0:
            output.append('Fake')
        else:
            output.append('Real')
        print("Image no.",y_test[i], output[-1])
        i=i+1
