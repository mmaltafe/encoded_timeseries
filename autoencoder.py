import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("INFO")
tf.autograph.set_verbosity(0)

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import glob
import pickle as pk

import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def d3_d2(mat):
    import numpy as np

    mat = np.array(mat)
    n, m, l = mat.shape
    return mat.reshape(n * m , l)

def argsDictionary(var_dict):

    structure_name = []

    for var in var_dict:
        if not var == "file":
            hp_n_values = str(var) + "-" * 2 + str(var_dict[var])
            structure_name.append(hp_n_values)

    structure_name = ("_" * 2).join(structure_name)

    return var_dict, structure_name


try:
    results_df = pd.read_csv("Results/results.csv")
except:
    results_df = pd.DataFrame()


input_list = glob.glob('sliding_window_data/*')

for input_file in input_list:
    # Fixed parameters

    args = {
        "nb_epoch": 200,
        "encoding_dim": 512,
        "hidden_dim_1": 256,
        "learning_rate": 0.001,
        "batch_size": 64,
        "input": input_file.split('.p')[0]
    }

    ##### Creates structure name #####
    args, struct_name = argsDictionary(args)

    with open(input_file, "rb") as f:
        data = pk.load(f)

    L, W = data.shape
    n_measures = int(data[:,1].max())
    n_ids = L//n_measures
    n_features = W - 2
    data = data[:,2:]

    data = data.reshape(n_ids, n_measures, n_features)

    train_data, test_data = train_test_split(data, test_size=0.25)

    ###### Creates NN structure #####

    # Setup network
    # make inputs

    autoencoder = tf.keras.Sequential()
    autoencoder.add(
        LSTM(
            args["encoding_dim"],
            activation="sigmoid",
            input_shape=(n_measures, n_features),
            return_sequences=True,
        )
    )
    autoencoder.add(Dropout(0.2))
    autoencoder.add(LSTM(args["hidden_dim_1"], activation="sigmoid", return_sequences=False))
    autoencoder.add(RepeatVector(n_measures)) 
    autoencoder.add(LSTM(args["hidden_dim_1"], activation="sigmoid", return_sequences=True))
    autoencoder.add(Dropout(0.2))
    autoencoder.add(
        LSTM(args["encoding_dim"], activation="sigmoid", return_sequences=True)
    )
    autoencoder.add(TimeDistributed(Dense(n_features)))
    autoencoder.compile(optimizer="adam", loss="mse")

    # Defining early stop

    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath="autoencoder_fraud.h5",
        mode="min",
        monitor="loss",
        verbose=0,
        save_best_only=True,
    )
    # define our early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.0001,
        patience=10,
        verbose=0,
        mode="min",
        restore_best_weights=True,
    )

    # Compiling NN

    opt = Adam(learning_rate=args["learning_rate"])

    autoencoder.compile(loss="mean_squared_error", optimizer=opt)

    autoencoder.summary()

    # break

    # Training
    # try:

    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=args["nb_epoch"],
        batch_size=args["batch_size"],
        shuffle=True,
        validation_data=(test_data, test_data),
        verbose=0,
        callbacks=[cp, early_stop],
    ).history

    # Ploting Model Loss

    fig, ax = plt.subplots()
    plt.plot(history["loss"], linewidth=2, label="Train")
    plt.plot(history["val_loss"], linewidth=2, label="Validation")
    plt.legend(loc="upper right")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    fig.savefig("Figures/model-loss__" + struct_name + "__.png", bbox_inches="tight")

    # Predicting Test values

    # start = datetime.now()

    test_x_predictions = autoencoder.predict(test_data)

    # end = datetime.now()

    # Calculating MSE
    mse = np.mean(np.power(d3_d2(test_data) - d3_d2(test_x_predictions), 2))

    args["Reconstruction_error"] = mse
    results_df = results_df.append(args, ignore_index=True)


    results_df.to_csv("Results/results.csv", index=None)

    ##### rodar com só uma janela
    break