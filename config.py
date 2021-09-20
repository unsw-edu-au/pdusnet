import os
import pandas as pd

# General settings
dataset_output_path = 'data/'
augmented_dataset_output_path = 'augmented_data/'
log_path = 'logs/'
results_path = 'results/'

data_path_prefix = '/mnt/raid/julie/nnfw_2/PlacentaData2018and2019_v51_6Nov2020/'


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
batch_size = 20
epochs = 50
n_filters = [8, 16, 32, 64, 128]
#n_filters = [16, 32, 64, 128, 256]

###################
my_nnfw_data_folder = '/mnt/raid/julie/nnfw_2/PlacentaData2018and2019_v51_6Nov2020_info/'




my_train_csv_file = '{}info_training_images.csv'.format(my_nnfw_data_folder)
my_validation_csv_file = '{}info_validation_images.csv'.format(my_nnfw_data_folder)
my_test_csv_file = '{}info_test_images.csv'.format(my_nnfw_data_folder)
df_train = pd.read_csv(my_train_csv_file)
df_validation = pd.read_csv(my_validation_csv_file)
df_test = pd.read_csv(my_test_csv_file)
###################

#train_samples = 60
#train_samples = len(df_train) #420 (approx. 60% of 705) - comibined 2018 & 2019 Placenta data
#train_samples = len(df_train) #160 (approx. 60% of 276) - 2018 Placenta data
#train_samples = len(df_train) #120 (approx. 60% of 226) - comibined 2018 & 2019 Placenta data_ set 1 (9x9 common patch in middle of middle slice)
#train_samples = len(df_train) #240 (approx. 60% of 417) - comibined 2018 & 2019 Placenta data randomized
train_samples = len(df_train) #240 (approx. 60% of 417) - v5 dataset

train_samples_total = 14 * train_samples

#validation_samples = 20
#validation_samples = 80
#validation_samples = len(df_validation) #120 (approx. 20% of 705) - comibined 2018 & 2019 Placenta data
#validation_samples = len(df_validation) #40 (approx. 20% of 276) - 2018 Placenta data
#validation_samples = len(df_validation) #40 (approx. 20% of 226) - comibined 2018 & 2019 Placenta data_ set 1 (9x9 common patch in middle of middle slice)
#validation_samples = len(df_validation) #80 (approx. 20% of 417) - comibined 2018 & 2019 Placenta data randomized
validation_samples = len(df_validation) #80 (approx. 20% of 417) - v5 dataset

#test_samples = 10
#test_samples = 20
#test_samples = 40
#test_samples = 80
#test_samples = len(df_test) #140 (approx. 20% of 705) - comibined 2018 & 2019 Placenta data
#test_samples = len(df_test) #60 (approx. 20% of 276) - 2018 Placenta data
#test_samples = len(df_test) #40 (approx. 20% of 226) - comibined 2018 & 2019 Placenta data_ set 1 (9x9 common patch in middle of middle slice)
#test_samples = len(df_test) #80 (approx. 20% of 417) - comibined 2018 & 2019 Placenta data randomized
#test_samples = len(df_test) #80 (approx. 20% of 417) - v5 dataset
#test_samples = len(df_test) #10 or 20 or 40 or 80 - v5 dataset variants
test_samples = 20




# Model settings
learning_rate = 1e-3
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08
decay = 0.000000199
dropout_rate = 0.25
#dropout_rate = 0.0
#dropout_rate = 0.5

# UNet++ settings
deep_supervision = True
