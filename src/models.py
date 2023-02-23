import os
import numpy as np

from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.regularizers import l1_l2
from keras.constraints import max_norm

# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.keras.optimizers import Adadelta
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.constraints import max_norm
# from keras.utils import multi_gpu_model
# from tensorflow.keras.utils import multi_gpu_model

from sklearn import metrics


BIN_LENGTH = 200
INPUT_LENGTH = 2000
EPOCH = 200
BATCH_SIZE = 64
GPUS = 4
MAX_NORM = 1


def define_model():

    model = Sequential()
    model.add(Conv1D(input_shape=(1924, 1),
                     filters=64,
                     kernel_size=4,
                     strides=1,
                     activation="relu",
                     kernel_regularizer=l1_l2(0.00001, 0.001),
                     kernel_constraint=max_norm(MAX_NORM)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(Conv1D(input_shape=(None, 64),
                     filters=128,
                     kernel_size=2,
                     strides=1,
                     activation="relu",
                     kernel_regularizer=l1_l2(0.00001, 0.001),
                     kernel_constraint=max_norm(MAX_NORM)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    return model



def run_model(data, save_dir, gpus=1):

    model = define_model()
    weights_file = os.path.join(save_dir, "weights.hdf5")
    model_file = os.path.join(save_dir, "model.hdf5")

    model.save(model_file)

    # Adadelta is recommended to be used with default values
    opt = Adadelta()

    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    X_train = data["train_data"]
    Y_train = data["train_labels"]
    X_validation = data["val_data"]
    Y_validation = data["val_labels"]
    X_test = data["test_data"]
    Y_test = data["test_labels"]

    _callbacks = []
    checkpoint = ModelCheckpoint(filepath=weights_file, save_best_only=True)
    _callbacks.append(checkpoint)
    earlystopping = EarlyStopping(monitor="val_loss", patience=15)
    _callbacks.append(earlystopping)

    model.fit(X_train,
               Y_train,
               batch_size=BATCH_SIZE * GPUS,
               epochs=EPOCH,
               validation_data=(X_validation, Y_validation),
               shuffle="batch",
               callbacks=_callbacks)

    Y_pred = model.predict(X_test, batch_size=BATCH_SIZE*GPUS)

    auc = metrics.roc_auc_score(Y_test, Y_pred)

    with open(os.path.join(save_dir, "auc.txt"), "w") as of:
        of.write("AUC: %f\n" % auc)

    [fprs, tprs, thrs] = metrics.roc_curve(Y_test, Y_pred[:, 0])

    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thr = thrs[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thr = thrs[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thr = thrs[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thr = thrs[sort_ix[0]]

    with open(os.path.join(save_dir, "fpr_threshold_scores.txt"), "w") as of:
        of.write("10 \t %f\n" % fpr10_thr)
        of.write("5 \t %f\n" % fpr5_thr)
        of.write("3 \t %f\n" % fpr3_thr)
        of.write("1 \t %f\n" % fpr1_thr)

    with open(os.path.join(save_dir, "roc_values.txt"), "w") as of:
        of.write("FPR\tTPR\tTHR\n")
        for fpr, tpr, thr in zip(fprs, tprs, thrs):
            of.write("%f\t%f\t%f\n" % (fpr, tpr, thr))


def get_models(phase_one_file, phase_two_file):

    # phase_one_weights_file = "../data/phase_one_weights.hdf5"
    # phase_two_weights_file = "../data/phase_two_weights.hdf5"

    model_1 = load_model(phase_one_file)
    model_2 = load_model(phase_two_file)

    return [model_1, model_2]


def get_phase_one_model(weights_file):

    model = load_model(weights_file)

    return model


def run_two_phases(X, models):

    Y_1 = models[0].predict(X)
    Y_1 = Y_1[..., np.newaxis]
    Y_2 = models[1].predict(Y_1)

    return Y_2
