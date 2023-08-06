import argparse
import os
import sys
import h5py
import numpy

from src import models


parser = argparse.ArgumentParser()

# I/O logistics.
parser.add_argument('-data-file',
                    dest="data_file",
                    default="data/phase_two_dataset.hdf5",
                    help="Dataset file created by data_prep.py")
parser.add_argument('-save-dir',
                    dest="save_dir",
                    default="trained_model",
                    help="Directory where the files will be saved. Will exit if the directory exists.")

# Model training parameters
parser.add_argument('-from-memory',
                    action='store_true',
                    help="Load the dataset to memory. Default: read directly from h5 file.")
parser.add_argument('-gpus',
                    dest="gpus",
                    default=1,
                    help="Number of GPU nodes to use for training.")
parser.add_argument('-opt',
                    dest="opt",
                    type=int,
                    default=1,
                    choices=[1, 2, 3, 4, 5],
                    help="Optimizer to use. 1: Adadelta (default), "
                                           "2: RMSProp, "
                                           "3: SGD, "
                                           "4: SGD with Nesterov momentum, "
                                           "5: Adam")
parser.add_argument('-filters-1',
                    dest="filters_1",
                    type=int,
                    default=64,
                    help="Number of convolutional filters (layer 1)")
parser.add_argument('-filters-2',
                    dest="filters_2",
                    type=int,
                    default=124,
                    help="Number of convolutional filters (layer 2)")
parser.add_argument('-do',
                    dest="do",
                    type=float,
                    default=0.4,
                    help="Dropout probability")
parser.add_argument('-max-norm',
                    dest="max_norm",
                    type=float,
                    default=1,
                    help="Max norm kernel constraint")
parser.add_argument('-l1',
                    dest="l1",
                    type=float,
                    default=0.0001,
                    help="L1 kernel regularizer")
parser.add_argument('-l2',
                    dest="l2",
                    type=float,
                    default=0.001,
                    help="L2 kernel regularizer")


# TO DELETE
parser.add_argument('-model-file',
                    action='store_true')

args = parser.parse_args()

if len(sys.argv) < 3:
    parser.print_help()
    sys.exit()

if not os.path.exists(args.data_file):
    print("Data file not found: ", args.data_file)
    sys.exit()

dir_append = ["f1", args.filters_1,
              "f2", args.filters_2,
              "do", args.do,
              "l1", int(-1 * numpy.log10(args.l1)),
              "l2", int(-1 * numpy.log10(args.l2)),
              "opt", args.opt]

dir_append = "__".join([str(_) for _ in dir_append])

save_dir = args.save_dir + "_" + dir_append

if os.path.exists(save_dir):
    print("Directory already exists: ", save_dir)
    print("Choose another destination.")
    sys.exit()
else:
    os.mkdir(save_dir)

# if os.path.exists(args.save_dir):
#     print("Directory already exists: ", args.save_dir)
#     print("Choose another destination.")
#     sys.exit()
# else:
#     os.mkdir(args.save_dir)


print("\n\n")
print("-----   Starting training with the parameters: -------")
for arg, value in vars(args).items():
    print("%s: %s" % (arg, value))
print("------------------------------------------------------")
print("\n\n")


with h5py.File(args.data_file, "r") as data:
    # models.hyperparameter_search(data, save_dir, opt)

    if args.from_memory:
        print("Loading the data file to memory. This may cause the job crash if the memory limit exceeds!\n\n")
        data = {k: data[k][()] for k in data.keys()}

    # if args.model_file:
    #     f = "/data/Dcode/jaya/two_phase_model/source_files/phase_two_model.hdf5"
    #     print("Loading the model description from:")
    #     print(f)
    #     print("\n")
    #     model = models.load_model(f)
    # else:
    #     model = None

    model = models.define_model_args(args)

    models.run_model(data=data, save_dir=save_dir, gpus=args.gpus, model=model, optimizer_no=args.opt)


print("Training completed!")