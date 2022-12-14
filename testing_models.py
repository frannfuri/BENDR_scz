import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import OrderedDict
from architectures import BENDRClassification
from datasets import standardDataset, charge_all_data

fold_nb = '2'
trained_name = 'decomp_study_ls1e-4bs16_deltaPANSS_posit'
#####################################################
# dataset parameters
directory = './datasets/decomp_study'
dmax = 0.000115710645
dmin = -0.000148340289
chns_consider = ['Cz', 'Pz']
target_f = 'deltaPANSS_posit'
apply_winsor = False
samples_length = 30
overlap_len = 20
size_output = 1
reg_option = True

# Load the previously trained weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_weights = torch.load('./results_new/best_model_f{}_{}.pt'.format(fold_nb, trained_name), map_location=device)


new_model_weights = {}
for key in model_weights.keys():
    new_model_weights[key[7:]] = model_weights[key]

# Load the data
array_epochs_all_subjects = charge_all_data(directory = directory,
                                                format_type = 'set',
                                                tlen = samples_length, overlap = overlap_len,
                                                data_max = dmax,
                                                data_min = dmin,
                                                chns_consider = chns_consider,
                                                labels_path = './datasets/labels',
                                                target_f = target_f,
                                                apply_winsor = apply_winsor)
is_first_rec = True
for rec in array_epochs_all_subjects:
    if is_first_rec:
        all_X = rec[0]
        all_y = rec[1]
        is_first_rec = False
    else:
        all_X = torch.cat((all_X, rec[0]), dim=0)
        all_y = torch.cat((all_y, rec[1]), dim=0)
all_dataset = standardDataset(all_X, all_y)

# IDs of test samples for the respective cross-validation iteration
test_ids = pd.read_csv('./logs_new/test_ids_{}_{}.csv'.format(fold_nb, trained_name), header=None).astype(int).to_numpy().squeeze()
test_subset = all_dataset[test_ids]

# Load model architecture
model = BENDRClassification(targets=size_output, samples_len=samples_length*256, n_chn=20,
                                        encoder_h=512,
                                        contextualizer_hidden=3076, projection_head=False,
                                        new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                        keep_layers=None,
                                        mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                        multi_gpu=False, return_features=True, regression_option=reg_option)
# Load the trained weigths
model.load_state_dict(new_model_weights)
model.eval()

model = model.to(device)
standard_test_set = standardDataset(test_subset[0], test_subset[1].to(int))
testloader = torch.utils.data.DataLoader(standard_test_set, batch_size=64)
criterion = torch.nn.MSELoss().to(device)

running_loss = 0.0
test_log = list()
test_num_samples = 0
it = 0
test_losses = []
for inputs, labels in testloader:
        print('iteration nb {}'.format(it))
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            test_metrics = OrderedDict()
            outputs = model(inputs)
            loss = criterion(outputs[0].squeeze(), labels)
            test_metrics['iter'] = it
            it += 1
            test_metrics['loss'] = loss.item()
            test_log.append(test_metrics)
            test_num_samples += 1*inputs.size(0)
        running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss
        test_losses.append(epoch_loss/test_num_samples)
        print('Loss: {:.4f} * numb. of samples, or Loss: {:.4f}.'.format(
            epoch_loss, epoch_loss/inputs.size(0)))
print('ready')

a = 0