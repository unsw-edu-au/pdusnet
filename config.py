import os
import pandas as pd

# General settings
dataset_output_path = 'data/'
augmented_dataset_output_path = 'augmented_data/'
log_path = 'logs/'
results_path = 'results/'
# data_path_prefix = '/mnt/raid/nipuna/AugmentedTrainingData64/'
#data_path_prefix = '/mnt/raid/nipuna/TrainingData64PD/'
# data_path_prefix = '/mnt/raid/nipuna/TrainingData64/'
#data_path_prefix = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018/'
#data_path_prefix = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018_nnfw/'
#data_path_prefix = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v2/'
#data_path_prefix = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets/'
#data_path_prefix = '/mnt/raid/mahmood/PlacentaData2018_all_in_one_train_validation_test/'
#data_path_prefix = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018_set_1/'
#data_path_prefix = '/mnt/raid/mahmood/PlacentaData2018and2019_randomized_all_in_one_train_validation_test_sets_updated_27Oct2020/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v21/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v22/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v23/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v24/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v25/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v26/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v27/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v31/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v32/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v33/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v34/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v35/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v36/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v37/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_triplets_removed_all_in_one_train_validation_test_all_multiples_of_20_v5_randomized_sets_updated_6Nov2020/'

#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5101/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5102/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5103/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5104/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5105/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5106/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5107/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5108/'

#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5201/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5202/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5203/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5204/'

#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5401/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5402/'

#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5801/'

data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v51_6Nov2020/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v52_17Nov2020/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v53_17Nov2020/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v54_17Nov2020/'
#data_path_prefix = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v55_17Nov2020/'



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
#my_nnfw_data_folder = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v2_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/PlacentaData2018_all_in_one_train_validation_test_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/PlacentaDataCombined2019and2018_set_1_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/PlacentaData2018and2019_randomized_all_in_one_train_validation_test_sets_updated_27Oct2020_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v21_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v22_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v23_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v24_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v25_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v26_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v27_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v31_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v32_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v33_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v34_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v35_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v36_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaDataCombined2019and2018_all_in_one_train_validation_test_v3_randomized_sets_v37_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_triplets_removed_all_in_one_train_validation_test_all_multiples_of_20_v5_randomized_sets_updated_6Nov2020_info/'

#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5101_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5102_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5103_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5104_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5105_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5106_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5107_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5108_info/'

#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5201_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5202_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5203_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5204_info/'

#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5401_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5402_info/'

#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v5_subset_v5801_info/'

my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v51_6Nov2020_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v52_17Nov2020_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v53_17Nov2020_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v54_17Nov2020_info/'
#my_nnfw_data_folder = '/mnt/raid/mahmood/DataSets/PlacentaData2018and2019_v55_17Nov2020_info/'



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
