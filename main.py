import os
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
import argparse

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import secrets
import odir
from odir_advance_plotting import Plotter
from odir_kappa_score import FinalScore
from odir_predictions_writer import Prediction
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import numpy as np

def main(config):

    batch_size = config.batch_size 
    num_classes = config.num_classes
    epochs = config.epochs
    patience = config.patience
    folder = config.folder
    filenpy = config.file_npy
    filenpylabels = config.file_npy_labels
    fileloadmodel = config.file_load_model
    num_images = config.num_images

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    class_weight = {0: 1.,
                    1: 1.583802025,
                    2: 8.996805112,
                    3: 10.24,
                    4: 10.05714286,
                    5: 1.,
                    6: 1.,
                    7: 2.505338078}

    token = secrets.token_hex(2)
    new_folder = os.path.join(folder, token)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    def load_data_test( filenpy,filenpylabels,challenge = 0):
        """Loads the ODIR dataset.

        Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

        """

        if challenge == 0:
            x_test = np.load(filenpy)
            y_test = np.load(filenpylabels)
        
        return (x_test, y_test)


    

    base_model = resnet50.ResNet50

    base_model = base_model(weights='imagenet', include_top=False)

    # Comment this out if you want to train all layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)
    model = load_model(fileloadmodel)
    model.summary()



    defined_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
    ]



    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=defined_metrics)

    (x_test, y_test) = load_data_test(filenpy,filenpylabels)

    x_test_drawing = x_test


    x_test = resnet50.preprocess_input(x_test)

    class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

    # plot data input
    plotter = Plotter(class_names)



    # test a prediction
    test_predictions_baseline = model.predict(x_test)
    print("plotting confusion matrix")
    plotter.plot_confusion_matrix_generic(y_test, test_predictions_baseline, os.path.join(new_folder, 'matrizC.png'), 0)

    # save the predictions
    prediction_writer = Prediction(test_predictions_baseline, num_images, new_folder)
    prediction_writer.save()
    prediction_writer.save_all(y_test)

    # show the final score
    score = FinalScore(new_folder)
    score.output()

    # plot output results
    plotter.plot_output(test_predictions_baseline, y_test, x_test_drawing, os.path.join(new_folder, 'OutputResults.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_classes', type=int, default=8, help='number of classs ')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--patience', type=int, default=8, help='mini-batch size')
    parser.add_argument('--folder',type=str, default='/media/user_home2/ynsuarez/AML/Proyecti/ODIR/ocular-disease-intelligent-recognition-deep-learning/Results')
    parser.add_argument('--file_npy',type=str,default='odir_testing_224.npy')
    parser.add_argument('--file_npy_labels',type=str,default='odir_testing_labels_224.npy')
    parser.add_argument('--file_load_model',type=str,default='model_weights.h5')
    parser.add_argument('--num_images',type = int, default = 400)

    config = parser.parse_args()
    print(config)
    main(config)