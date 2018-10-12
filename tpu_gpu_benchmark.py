import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import keras_support
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.datasets import cifar100
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, History
import tensorflow.keras.backend as K
import numpy as np
import os, time, pickle

class Timer(Callback):
    def __init__(self):
        self.inital_time_starts = time.time()
        
    def on_train_begin(self, logs):
        self.inital_time = time.time() - self.inital_time_starts
        self.epoch_starts = time.time()
        self.times = []

    def on_epoch_end(self, epoch, logs):
        self.times.append(time.time()-self.epoch_starts)
        self.epoch_starts = time.time()

def create_residual_blocks(input_tensor, base_ch, k, N):
    start_tensor = input_tensor
    for i in range(N):
        x = Conv2D(base_ch*k, 7, padding="same")(start_tensor)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(base_ch*k, 7, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Add()([start_tensor, x])
        start_tensor = x
    return x

# WideResNet
def create_wideresnet(k, N, use_tpu):
    input = Input(shape=(32, 32, 3))
    # conv1 : 32x32
    x = Conv2D(16*k, 1)(input)
    x = create_residual_blocks(x, 16, k, N)
    # downsampling 32->16
    x = AveragePooling2D(2)(x)
    x = Conv2D(32*k, 1)(x)
    # conv2 : 16x16
    x = create_residual_blocks(x, 32, k, N)
    # downsampling 16->8
    x = AveragePooling2D(2)(x)
    x = Conv2D(64*k, 1)(x)
    # conv4 : 8x8
    x = create_residual_blocks(x, 64, k, N)
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation="softmax")(x)

    model = Model(input, x)
    model.compile(Adam(), loss="categorical_crossentropy", metrics=["acc"])

    if use_tpu:
        tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
        strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
        model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
    return model

def single_trial(use_tpu, batch_size, use_validation, use_augment, from_storage, parallel_workers):
    K.clear_session()
    model = create_wideresnet(7, 4, use_tpu)

    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        width_shift_range=4.0/32,
        height_shift_range=4.0/32,
        horizontal_flip=True)
    val_gen = ImageDataGenerator(
        rescale=1.0/255)

    if not from_storage:
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        if not use_augment:
            X_train = (X_train / 255.0).astype(np.float32)
            X_test = (X_test / 255.0).astype(np.float32)

    timer = Timer()
    hist = History()

    n_train_examples, n_test_examples = 50000, 10000
    n_epochs = 1
    multiprocess = False if parallel_workers <= 1 else True

    print("Start training...")
    print(f"use_tpu:{use_tpu}, batch_size:{batch_size}, use_validation:{use_validation}, use_augment:{use_augment}, from_storage:{from_storage}, workers:{parallel_workers}")

    if from_storage:
        if use_augment:
            if use_validation:
                model.fit_generator(train_gen.flow_from_directory("cifar100-raw/train", target_size=(32, 32), 
                                                                  class_mode="categorical", shuffle=True,
                                                                  batch_size=batch_size), 
                                    steps_per_epoch=n_train_examples//batch_size, epochs=n_epochs,
                                    callbacks=[timer, hist],
                                    workers=parallel_workers, use_multiprocessing=multiprocess,
                                    validation_data=val_gen.flow_from_directory("cifar100-raw/test", target_size=(32, 32),
                                                                                class_mode="categorical", shuffle=True,
                                                                                batch_size=batch_size),
                                    validation_steps=n_test_examples//batch_size)
            else:
                model.fit_generator(train_gen.flow_from_directory("cifar100-raw/train", target_size=(32, 32), 
                                                                  class_mode="categorical", shuffle=True,
                                                                  batch_size=batch_size), 
                                    steps_per_epoch=n_train_examples//batch_size, epochs=n_epochs,
                                    callbacks=[timer, hist],
                                    workers=parallel_workers, use_multiprocessing=multiprocess)
        else:
            if use_validation:
                model.fit_generator(val_gen.flow_from_directory("cifar100-raw/train", target_size=(32, 32),
                                                                class_mode="categorical", shuffle=True,
                                                                batch_size=batch_size),
                                    steps_per_epoch=n_train_examples//batch_size, epochs=n_epochs,
                                    callbacks=[timer, hist],
                                    workers=parallel_workers, use_multiprocessing=multiprocess,
                                    validation_data=val_gen.flow_from_directory("cifar100-raw/test", target_size=(32, 32),
                                                                                class_mode="categorical", shuffle=True,
                                                                                batch_size=batch_size),
                                    validation_steps=n_test_examples//batch_size)
            else:
                model.fit_generator(val_gen.flow_from_directory("cifar100-raw/train", target_size=(32, 32),
                                                                class_mode="categorical", shuffle=True,
                                                                batch_size=batch_size),
                                    steps_per_epoch=n_train_examples//batch_size, epochs=n_epochs,
                                    callbacks=[timer, hist],
                                    workers=parallel_workers, use_multiprocessing=multiprocess)
    else:
        if use_augment:
            if use_validation:
                model.fit_generator(train_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
                                    steps_per_epoch=n_train_examples//batch_size,
                                    epochs=n_epochs, callbacks=[timer, hist],
                                    workers=parallel_workers, use_multiprocessing=multiprocess,
                                    validation_data=val_gen.flow(X_test, y_test), validation_steps=n_test_examples//batch_size)
            else:
                model.fit_generator(train_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
                                    steps_per_epoch=n_train_examples//batch_size,
                                    epochs=n_epochs, callbacks=[timer, hist],
                                    workers=parallel_workers, use_multiprocessing=multiprocess)
        else:
            # fitは並列化できない
            if use_validation:
                model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, callbacks=[timer, hist],
                          validation_data=(X_test, y_test))
            else:
                model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, callbacks=[timer, hist])

    history = hist.history
    history["initial_time"] = timer.inital_time
    history["times"] = timer.times

    result = {
        "device": "tpu" if use_tpu else "gpu",
        "batch_size" : batch_size,
        "use_validation" : use_validation,
        "use_augmentation" : use_augment,
        "from_storage": from_storage,
        "result" : history,
        "num_workers" : parallel_workers
    }

    return result

def trial(use_tpu, batch_size, separate_mode=-1):
    flag = "tpu" if use_tpu else "gpu"
    if separate_mode == -1:
        filename = f"{flag}_batchsize_{batch_size}.dat"
    else:
        filename = f"{flag}_batchsize_{batch_size}_sep{separate_mode}.dat"
    result = []

    if separate_mode in [-1, 0]:
        result.append(single_trial(use_tpu, batch_size, use_validation=False, use_augment=False, from_storage=False, parallel_workers=1))
        result.append(single_trial(use_tpu, batch_size, use_validation=True, use_augment=False, from_storage=False, parallel_workers=1))
    if separate_mode in [-1, 1]:
        result.append(single_trial(use_tpu, batch_size, use_validation=True, use_augment=True, from_storage=False, parallel_workers=1))
        result.append(single_trial(use_tpu, batch_size, use_validation=False, use_augment=False, from_storage=True, parallel_workers=1))
    if separate_mode in [-1, 2]:
        result.append(single_trial(use_tpu, batch_size, use_validation=True, use_augment=False, from_storage=True, parallel_workers=1))
        result.append(single_trial(use_tpu, batch_size, use_validation=True, use_augment=True, from_storage=True, parallel_workers=1))
    if separate_mode in [-1, 3]:
        result.append(single_trial(use_tpu, batch_size, use_validation=False, use_augment=False, from_storage=True, parallel_workers=4))
        result.append(single_trial(use_tpu, batch_size, use_validation=True, use_augment=True, from_storage=True, parallel_workers=4))

    with open(filename, "wb") as fp:
        pickle.dump(result, fp)
    return filename

def appendix_trial(batch_size, use_tpu=True, sep=-1):
    tpu_flag = "tpu" if use_tpu else "gpu"
    filename = f"appendix_{tpu_flag}_batch_size_{batch_size}"
    if sep >= 0: filename += f"_sep_{sep}"
    filename += ".dat"

    result = {}

    for mode in range(3):
        if sep >= 0:
            if sep != mode: continue
        K.clear_session()
        model = create_wideresnet(7, 4, use_tpu)

        # mode 1 = そのままfit
        # mode 2 = バッチサイズの倍数に切り詰めてfit
        # mode 3 = fit_generator
        data_gen = ImageDataGenerator(rescale=1.0/255)

        nb_epochs = 20
        (X_train, y_train), (_, _) = cifar100.load_data()

        timer = Timer()
        hist = History()

        print("Start training...")
        print("mode = ", mode)

        if mode == 0:
            X_train = X_train / 255.0
            y_train = to_categorical(y_train)
            model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=[timer, hist])
        elif mode == 1:
            n_train = (X_train.shape[0] // batch_size) * batch_size
            X_train = X_train[:n_train, :, :, :] / 255.0
            y_train = to_categorical(y_train[:n_train, :])
            model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, callbacks=[timer, hist])
        elif mode == 2:
            y_train = to_categorical(y_train)
            steps_per_epoch = X_train.shape[0] // batch_size
            model.fit_generator(data_gen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
                                steps_per_epoch=steps_per_epoch, epochs=nb_epochs, callbacks=[timer, hist])

        history = hist.history
        history["initial_time"] = timer.inital_time
        history["times"] = timer.times
        result[mode] = history

    with open(filename, "wb") as fp:
        pickle.dump(result, fp)
    return filename


if __name__ == "__main__":
    filename = trial(False, 256, 2) # True if use TPU
    #filename = appendix_trial(4096, sep=0)