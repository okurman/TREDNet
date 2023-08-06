import argparse
import os.path
import sys
from os.path import join

import h5py
import numpy as np
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.constraints import max_norm
from keras.regularizers import L1L2
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.layers import Convolution1D
from keras.layers.core import Flatten


from sklearn.utils import class_weight
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


np.random.seed(12345)


def create_model():

    model = Sequential()
    model.add(Convolution1D(256, 1, activation='relu', input_shape=(220, 1),
                            kernel_regularizer=L1L2(l1=1e-4, l2=1e-3), kernel_constraint=max_norm(1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0))
    model.add(Dropout(0.2))
    model.add(Convolution1D(256, 1, activation='relu', kernel_regularizer=L1L2(l1=1e-4, l2=1e-3),
                            kernel_constraint=max_norm(1)))
    model.add(LeakyReLU(0))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='ADAM', metrics=['accuracy'])

    return model


def train_model(dataset_file, save_dir):

    dataset = h5py.File(dataset_file, "r")
    X_train = dataset["X_train"][()]
    Y_train = dataset["Y_train"][()]
    X_test = dataset["X_test"][()]
    Y_test = dataset["Y_test"][()]

    print("Launching model training.")
    model = create_model()
    model.save(join(save_dir, "model.hdf5"))

    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    checkpointer = ModelCheckpoint(filepath=join(save_dir, "model_weights.hdf5"), verbose=1, save_best_only=True)
    history_log_file = os.path.join(save_dir, "training_history_log.tab")
    history_logger = CSVLogger(filename=history_log_file, separator="\t", append=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=30)

    history = model.fit(X_train, Y_train,
              epochs=500,
              batch_size=5000,
              shuffle=True,
              validation_split=0.2,
              callbacks=[checkpointer, earlystopper, history_logger],
              class_weight=class_weights,
              verbose=2)

    auroc, auprc = evaluate_model(X_test, Y_test, model, save_dir)

    plot_file = os.path.join(save_dir, "training_plot.pdf")
    title = "auROC: %.3f auPRC: %.3f" % (auroc, auprc)
    generate_plot_history(history, plot_file, title=title)


def evaluate_model(X_test, Y_test, model, save_dir):

    print("Testing model...", model.evaluate(X_test, Y_test))
    Y_pred = model.predict(X_test)

    auroc = metrics.roc_auc_score(Y_test, Y_pred)
    auprc = metrics.average_precision_score(Y_test, Y_pred, pos_label=1)

    with open(join(save_dir, "auROC_auPRC.txt"), "w") as of:
        of.write("auROC: %f\n" % auroc)
        of.write("auPRC: %f\n" % auprc)

    fprs, tprs, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
    with open(join(save_dir, "ROC_values.txt"), "w") as of:
        for _fpr, _tpr, _thr in zip(fprs, tprs, thresholds):
            out_line = "%f\t%f\t%s\n" % (_fpr, _tpr, _thr)
            of.write(out_line)

    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thr = thresholds[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thr = thresholds[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thr = thresholds[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thr = thresholds[sort_ix[0]]

    with open(os.path.join(save_dir, "fpr_threshold_scores.txt"), "w") as of:
        of.write("10 \t %s\n" % str(fpr10_thr))
        of.write("5 \t %s\n" % str(fpr5_thr))
        of.write("3 \t %s\n" % str(fpr3_thr))
        of.write("1 \t %s\n" % str(fpr1_thr))

    return auroc, auprc


def evaluate_model_standolone(data_file, save_dir):

    with h5py.File(data_file, "r") as data:

        X_test = data["X_test"]
        Y_test = data["Y_test"]

        model = load_model(os.path.join(save_dir, "model.hdf5"))
        model.load_weights(os.path.join(save_dir, "model_weights.hdf5"))

        print("Testing model...", model.evaluate(X_test, Y_test))
        Y_pred = model.predict(X_test)

        auroc = metrics.roc_auc_score(Y_test, Y_pred)
        auprc = metrics.average_precision_score(Y_test, Y_pred, pos_label=1)

        with open(join(save_dir, "auROC_auPRC.txt"), "w") as of:
            of.write("auROC: %f\n" % auroc)
            of.write("auPRC: %f\n" % auprc)

        fprs, tprs, thresholds = metrics.roc_curve(Y_test, Y_pred, pos_label=1)
        with open(join(save_dir, "ROC_values.txt"), "w") as of:
            for _fpr, _tpr, _thr in zip(fprs, tprs, thresholds):
                out_line = "%f\t%f\t%s\n" % (_fpr, _tpr, _thr)
                of.write(out_line)

        sort_ix = np.argsort(np.abs(fprs - 0.1))
        fpr10_thr = thresholds[sort_ix[0]]

        sort_ix = np.argsort(np.abs(fprs - 0.05))
        fpr5_thr = thresholds[sort_ix[0]]

        sort_ix = np.argsort(np.abs(fprs - 0.03))
        fpr3_thr = thresholds[sort_ix[0]]

        sort_ix = np.argsort(np.abs(fprs - 0.01))
        fpr1_thr = thresholds[sort_ix[0]]

        with open(os.path.join(save_dir, "fpr_threshold_scores.txt"), "w") as of:
            of.write("10 \t %s\n" % str(fpr10_thr))
            of.write("5 \t %s\n" % str(fpr5_thr))
            of.write("3 \t %s\n" % str(fpr3_thr))
            of.write("1 \t %s\n" % str(fpr1_thr))


def generate_plot_history(history, save_file, title=None):

    if type(history) == str:
        history = pd.read_table(history)
    else:
        history = history.history

    loss = history["loss"]
    val_loss = history["val_loss"]
    acc = history["acc"]
    val_acc = history["val_acc"]

    x_range = np.arange(len(loss))

    fig, axs = plt.subplots(1, 2)

    axs[0].plot(x_range, loss, label="train")
    axs[0].plot(x_range, val_loss, label="val")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(x_range, acc, label="train")
    axs[1].plot(x_range, val_acc, label="val")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_file)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset-file', dest="dataset_file", help="hdf5 file containing the dataset", required=True)
    parser.add_argument('-save-dir', dest="save_dir",
                        help="Directory where the trained model and weights will be saved.", required=True)

    args = parser.parse_args()

    dataset_file = args.dataset_file

    save_dir = args.save_dir

    train_model(dataset_file, save_dir)


