import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import OrderedDict
from architectures import BENDRClassification
from datasets import standardDataset, charge_all_data

random_weigths=False
fold_nb = '0'
results_folder = 'results_new'
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

different_dataset=False
n_tests = (37 if different_dataset else None)
n_records_other_ds = (144 if different_dataset else None)
name_output = 'mae'
##################################
# Load the previously trained weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_weights = torch.load('./{}/best_model_f{}_{}.pt'.format(results_folder, fold_nb, trained_name), map_location=device)


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
#all_dataset = standardDataset(all_X, all_y)

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
criterion = torch.nn.MSELoss().to(device)

all_X = all_X.to(device)
all_y = all_y.to(device)
with torch.no_grad():
    outputs = model(all_X)

torch.save(outputs[0], './outputs_{}.pt'.format(name_output))


##
loss_used = 'mae'
nb_of_windows_per_record = []
for rec_i in range(len(array_epochs_all_subjects)):
    nb_of_windows_per_record.append(len(array_epochs_all_subjects[rec_i][1]))
nb_of_windows_per_subj = [144, 148, 132, 105, 137]

rec_divisions = []
count = 0
for i in nb_of_windows_per_record:
    count += i
    rec_divisions.append(count)
subj_divisions = []
count = 0
for i in nb_of_windows_per_subj:
    count += i
    subj_divisions.append(count)

#outputs =

plt.figure(figsize=(10, 4))
plt.plot(all_y, '*', label='Label', alpha=0.8, markersize=3)
plt.plot(outputs, 'o', label='Predicted', alpha=0.8, markersize=3)
plt.title('Predicted label for the windows sorted in time\n[model trained w/all subjs.][All subjs. data][Loss: {}]'.format(loss_used))
for i in range(len(rec_divisions)):
    if i == (len(rec_divisions)-1):
        plt.axvline(rec_divisions[i], color='red', linestyle='dashed', linewidth=0.5, label='Day separation')
    else:
        plt.axvline(rec_divisions[i], color='red', linestyle='dashed', linewidth=0.5)
for j in range(len(subj_divisions)):
    if j == (len(subj_divisions)-1):
        plt.axvline(subj_divisions[j], color='black', linewidth=0.7, label='Subj. separation')
    else:
        plt.axvline(subj_divisions[j], color='black', linewidth=0.7)
plt.ylabel(r'$\Delta$' + ' {}'.format(target_f[5:]))
plt.legend()
plt.xlabel('Sorted windows in time')
plt.tight_layout()
plt.xlim((0-0.8, 666+0.8))
plt.ylim((-1, 15))

a = 0
