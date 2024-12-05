import os
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.optimizers import Adadelta, RMSprop, SGD, Adam
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.regularizers import l1_l2
from keras.constraints import max_norm
from keras.utils import multi_gpu_model

from sklearn import metrics

INPUT_LENGTH = 1924
EPOCH = 200
PATIENCE = 20

BATCH_SIZE = 64
MAX_NORM = 1
L1 = 0.0001
L2 = 0.001


def define_model_args(args, input_length=INPUT_LENGTH):

	model = Sequential()
	model.add(Conv1D(input_shape=(input_length, 1),
					 filters=args.filters_1,
					 kernel_size=4,
					 strides=1,
					 activation="relu",
					 kernel_regularizer=l1_l2(args.l1, args.l2),
					 kernel_constraint=max_norm(args.max_norm)))

	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(args.do))
	model.add(Conv1D(input_shape=(None, args.filters_1),
					 filters=args.filters_2,
					 kernel_size=2,
					 strides=1,
					 activation="relu",
					 kernel_regularizer=l1_l2(args.l1, args.l2),
					 kernel_constraint=max_norm(args.max_norm)))

	model.add(Dropout(args.do))
	model.add(Flatten())
	model.add(Dense(units=100, activation="relu"))
	model.add(Dense(units=50, activation="relu"))
	model.add(Dense(units=1, activation="sigmoid"))

	return model


def define_model(input_length=INPUT_LENGTH):

	model = Sequential()
	model.add(Conv1D(input_shape=(input_length, 1),
							filters=64,
							kernel_size=4,
							strides=1,
							activation="relu",
							kernel_regularizer=l1_l2(L1, L2),
							kernel_constraint=max_norm(MAX_NORM)))

	model.add(BatchNormalization())
	model.add(MaxPooling1D(pool_size=2))
	model.add(Dropout(0.4))
	model.add(Conv1D(input_shape=(None, 64),
					 filters=128,
					 kernel_size=2,
					 strides=1,
					 activation="relu",
					 kernel_regularizer=l1_l2(L1, L2),
					 kernel_constraint=max_norm(MAX_NORM)))

	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(units=100, activation="relu"))
	model.add(Dense(units=50, activation="relu"))
	model.add(Dense(units=1, activation="sigmoid"))

	return model


def get_optimizer(opt_choice):

	if opt_choice == 1:
		return Adadelta()
	elif opt_choice == 2:
		return RMSprop()
	elif opt_choice == 3:
		return SGD()
	elif opt_choice == 4:
		return SGD(nesterov=True)
	elif opt_choice == 5:
		return Adam()


def run_model(data, save_dir, gpus=1, model=None, optimizer_no=1):

	if model is None:
		input_len = data["X_train"].shape[1] if "X_train" in data else data["train_data"].shape[1]
		model = define_model(input_len)

	weights_file = os.path.join(save_dir, "weights.hdf5")
	model_file = os.path.join(save_dir, "model.hdf5")

	model.save(model_file)

	opt = get_optimizer(optimizer_no)

	if gpus > 1:
		model = multi_gpu_model(model, gpus=gpus)

	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

	if "X_train" in data:
		X_train = data["X_train"]
		Y_train = data["Y_train"]
		X_validation = data["X_val"]
		Y_validation = data["Y_val"]
		X_test = data["X_test"]
		Y_test = data["Y_test"]
	else:
		X_train = data["train_data"]
		Y_train = data["train_labels"]
		X_validation = data["val_data"]
		Y_validation = data["val_labels"]
		X_test = data["test_data"]
		Y_test = data["test_labels"]

	if len(X_train.shape) == 2:
		print("Loading the data to memory")
		X_train = X_train[()][..., np.newaxis]
		X_validation = X_validation[()][..., np.newaxis]
		X_test = X_test[()][..., np.newaxis]

	_callbacks = []
	checkpoint = ModelCheckpoint(filepath=weights_file, save_best_only=True, save_weights_only=True)
	_callbacks.append(checkpoint)
	earlystopping = EarlyStopping(monitor="val_loss", patience=PATIENCE)
	_callbacks.append(earlystopping)

	history_log_file = os.path.join(save_dir, "training_history_log.tab")
	history_logger = CSVLogger(filename=history_log_file, separator="\t", append=True)
	_callbacks.append(history_logger)

	shuffle_mode = True if type(data) is dict else "batch"

	history = model.fit(X_train,
						Y_train,
						batch_size=BATCH_SIZE * gpus,
						epochs=EPOCH,
						validation_data=(X_validation, Y_validation),
						shuffle=shuffle_mode,
						callbacks=_callbacks)

	Y_pred = model.predict(X_test, batch_size=BATCH_SIZE * gpus)

	auc = metrics.roc_auc_score(Y_test, Y_pred)
	prc = metrics.average_precision_score(Y_test, Y_pred)

	with open(os.path.join(save_dir, "auc.txt"), "w") as of:
		of.write("auROC: %f\n" % auc)
		of.write("auPRC: %f\n" % prc)

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

	plot_file = os.path.join(save_dir, "training_plot.pdf")
	title = "auROC: %.3f auPRC: %.3f" % (auc, prc)
	generate_plot_history(history, plot_file, title=title)


def evaluate(model, data, save_dir):

	if "X_train" in data:
		X_test = data["X_test"]
		Y_test = data["Y_test"]
	else:
		X_test = data["test_data"]
		Y_test = data["test_labels"]

	Y_pred = model.predict(X_test)

	auc = metrics.roc_auc_score(Y_test, Y_pred)
	prc = metrics.average_precision_score(Y_test, Y_pred)

	# with open(os.path.join(save_dir, "auc.txt"), "w") as of:
	# 	of.write("auROC: %f\n" % auc)
	# 	of.write("auPRC: %f\n" % prc)

	return auc, prc


