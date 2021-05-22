# Description: Train and evaluate CNNs for breast cancer classification

import os
import loader
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn import metrics

def show_examples(images, labels):
    class_names = ['benign', 'malignant']
    plt.figure(figsize=(10,10))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()

def baseline_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    return model

def multilayer_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    return model

def multilayer_regularized_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(.5))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(.5))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(.5))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(.5))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    return model

def deep_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(.5))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(.5))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Dropout(.5))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(.7))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(.7))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(.7))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(.7))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def evaluate_model(train, test, model, callback, name):
    train_images, train_labels = train
    test_images, test_labels = test
    print(f"Evaluating {name} model")
    model.compile(optimizer = 'adam', 
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
            metrics = ['accuracy'])
    history = model.fit(train_images, train_labels, epochs=25, validation_split=.1, callbacks=[callback])
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"Test Accuracy for {name}: {test_acc}")
    res = {'train_accuracy' : history.history['accuracy'], 'val_accuracy' : history.history['val_accuracy']}
    df = pd.DataFrame.from_dict(res)
    df.to_csv(f"results/{name}_results.csv", index=False, header=True)
    print("Model evaluation completed")
    return test_loss, test_acc

def compute_roc(model, name, test_images, test_labels):
    preds = model.predict(test_images)
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, preds)
    df = pd.DataFrame.from_dict({'fpr' : fpr, 'tpr' : tpr, 'threshold' : thresholds})
    df.to_csv(f"results/{name}_roc.csv", index=False, header=True)


def main():
    metadata_fp = 'image_metadata.csv'
    df = pd.read_csv(metadata_fp)
    # df = df.head(10)
    train, test = loader.load_data(df, .9)
    #(train_images, train_labels), (test_images, test_labels) = loader.load_data(df, .9)
    #print(train_images.shape)
    #print(train_labels.shape)
    baseline = baseline_model()
    multi_cnn = multilayer_cnn()
    multireg_cnn = multilayer_regularized_cnn()
    deepCNN  = deep_cnn()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)
  
    if not os.path.exists('results'):
        os.mkdir('results')



    baseline_res = evaluate_model(train, test, baseline, callback, "baseline")
    compute_roc(baseline, "baseline", *test)
    
    mcnn_res = evaluate_model(train, test, multi_cnn, callback, "multilayer_cnn")
    compute_roc(multi_cnn, "multilayer_cnn", *test)
    
    mrcnn_res = evaluate_model(train, test, multireg_cnn, callback, "multilayer_reg_cnn")
    compute_roc(multireg_cnn, "multilayer_reg_cnn", *test)
    
    dcnn_res = evaluate_model(train, test, deepCNN, callback, "deep_cnn")
    compute_roc(deepCNN, "deep_cnn", *test)
    
    res = [baseline_res, mcnn_res, mrcnn_res, dcnn_res]
    test_acc = {'name' : ['baseline', 'multilayer', 'multilayer_regularized', 'deep'], 'accuracy' : [t[1] for t in res]}
    df = pd.DataFrame.from_dict(test_acc)
    df.to_csv("results/test_accuracies.csv", index=False, header=True)


    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()

    #test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    #print(f"Test Accuracy: {test_acc}")

if __name__ == '__main__':
    main()
