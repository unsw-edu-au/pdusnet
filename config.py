# General settings
dataset_output_path = 'data/'
augmented_dataset_output_path = 'augmented_data/'
log_path = 'logs/'
results_path = 'results/'
# data_path_prefix = '/mnt/raid/nipuna/AugmentedTrainingData64/'
data_path_prefix = '/mnt/raid/nipuna/TrainingData64PD/'
# data_path_prefix = '/mnt/raid/nipuna/TrainingData64/'
checkpoint_path = 'checkpoints/'
csv_log_path = 'csv_logs/'
image_dir_path = 'model_images/'
pred_dir_path = 'preds/'

# Dataset
vol_x, vol_y, vol_z = 64, 64, 64
n_channels = 1

# Model
model_type = "unet++"

# Training settings
batch_size = 10
epochs = 30
# n_filters = [8, 16, 32, 64, 128]
n_filters = [16, 32, 64, 128, 256]

train_samples = 60
train_samples_total = 14 * train_samples
validation_samples = 20
test_samples = 20

# Model settings
learning_rate = 1e-3
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08
decay = 0.000000199
dropout_rate = 0.25

# UNet++ settings
deep_supervision = True
