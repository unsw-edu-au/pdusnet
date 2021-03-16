import os, time
from config import checkpoint_path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback, CSVLogger
from helpers import generate_checkpoint_path, generate_csv_log_path, generate_tensorboard_path


def get_callbacks(path_prefix):
    return [
        init_model_checkpoints(path_prefix),
        init_tensorboard(path_prefix),
        init_early_stop(),
        init_csv_logger(path_prefix)
    ]


def init_early_stop():
    return EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')


def init_csv_logger(path_prefix):
    filepath = generate_csv_log_path(path_prefix)
    return CSVLogger(filepath)


def init_model_checkpoints(path_prefix):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    filepath = generate_checkpoint_path(path_prefix)
    return ModelCheckpoint(
        filepath, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False,
        save_frequency=1)


def init_tensorboard(path_prefix):
    filepath = generate_tensorboard_path(path_prefix)
    return TensorBoard(log_dir=filepath, histogram_freq=1)


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
