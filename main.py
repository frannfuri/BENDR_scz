import argparse
import torch
import yaml
import os
import numpy as np
from datasets import charge_all_data, standardDataset
from architectures import MODEL_CHOICES, LinearHeadBENDR, BENDRClassification
from trainables import train_model
from torch.optim import lr_scheduler
from torch import nn
from sklearn.model_selection import KFold

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tunes BENDER models.")
    parser.add_argument('model', choices=MODEL_CHOICES)
    # TODO: parser.add_argument('--subject-specific', action='store_true', help="Fine-tune on target subject alone.")
    # TODO: parser.add_argument('--mdl', action='store_true', help="Fine-tune on target subject using all extra data.")
    parser.add_argument('--freeze-encoder', action='store_true', help="Whether to keep the encoder stage frozen. "
                                                                      "Will only be done if not randomly initialized.")
    parser.add_argument('--random-init', action='store_true', help='Randomly initialized BENDR for comparison.')
    # TODO:
    parser.add_argument('--multi-gpu', action='store_true', help='Distribute BENDR over multiple GPUs')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of dataloader workers.')
    parser.add_argument('--results-filename', default=None, help='What to name the spreadsheet produced with all '
                                                                 'final results.')
    parser.add_argument('--dataset-directory', default=None,
                        help='Where is the ubication of the data samples and the information '
                             'associated to them.')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('---Using ' + str(device) + 'device---')
    os.makedirs('./results_' + args.results_filename, exist_ok=True)
    os.makedirs('./logs_' + args.results_filename, exist_ok=True)

    with open(args.dataset_directory + '/info.yml') as infile:
        data_settings = yaml.load(infile, Loader=yaml.FullLoader)

    # Data sample params
    samples_tlen = data_settings['tlen']           # in seconds
    samples_overlap = data_settings['overlap_len'] # in seconds

    # Load dataset
    array_epochs_all_subjects = charge_all_data(directory=args.dataset_directory,
                                                format_type=data_settings['format_type'],
                                                tlen=samples_tlen, overlap=samples_overlap,
                                                data_max=data_settings['data_max'],
                                                data_min=data_settings['data_min'],
                                                chns_consider=data_settings['chns_to_consider'],
                                                labels_path=data_settings['labels_path'],
                                                target_f=data_settings['target_feature'],
                                                apply_winsor=data_settings['apply_winsorising'])
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

    # Set fixed random number seed
    torch.manual_seed(298)
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=data_settings['folds'], shuffle=True)

    # Train parameters
    lr = float(data_settings['lr'])
    num_epochs = data_settings['epochs']
    bs = data_settings['batch_size']
    ####################
    # Start print
    print('-------------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_dataset)):
        # Print
        print('FOLD n°{}'.format(fold))
        print('------------------------')
        np.savetxt('./logs_' + args.results_filename + '/train_ids_' + str(fold) + '.csv', train_ids, delimiter=',')
        np.savetxt('./logs_' + args.results_filename + '/test_ids_' + str(fold) + '.csv', test_ids, delimiter=',')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
            all_dataset,
            batch_size=bs, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
            all_dataset,
            batch_size=bs, sampler=test_subsampler)
        dataloaders = {'train': trainloader, 'val': testloader}

        dataset_sizes = {x: len(dataloaders[x]) * bs for x in ['train', 'val']}

        # MODEL
        if args.model == MODEL_CHOICES[0]:
            model = BENDRClassification(targets=data_settings['num_cls'], samples_len=samples_tlen * 256, n_chn=20,
                                        encoder_h=512,
                                        contextualizer_hidden=3076, projection_head=False,
                                        new_projection_layers=0, dropout=0., trial_embeddings=None, layer_drop=0,
                                        keep_layers=None,
                                        mask_p_t=0.01, mask_p_c=0.005, mask_t_span=0.1, mask_c_span=0.1,
                                        multi_gpu=False, return_features=True, regression_option=data_settings['regression_option'])
        else:
            model = LinearHeadBENDR(n_targets=data_settings['num_cls'], samples_len=samples_tlen * 256, n_chn=20,
                                    encoder_h=512,
                                    projection_head=False,
                                    enc_do=0.1, feat_do=0.4, pool_length=4, mask_p_t=0.01, mask_p_c=0.005,
                                    mask_t_span=0.05,
                                    mask_c_span=0.1, classifier_layers=1, return_features=True)

        # model = model.to(device)
        if not args.random_init:
            model.load_pretrained_modules('./datasets/encoder.pt', './datasets/contextualizer.pt',
                                          freeze_encoder=args.freeze_encoder)
        if args.multi_gpu:
            model = nn.DataParallel(model)
        model = model.to(device)
        if data_settings['regression_option']:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)
            criterion = torch.nn.MSELoss().to(device)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
            criterion = torch.nn.CrossEntropyLoss().to(device)
        sched = lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch=len(dataloaders['train']),
                                        pct_start=0.3, last_epoch=-1)

        best_model, acc_curves, loss_curves, train_log, valid_log = train_model(model, criterion, optimizer, sched,
                                                                                dataloaders, dataset_sizes, device,
                                                                                num_epochs)
        train_log.to_pickle("./logs_{}/train_log_f{}.pkl".format(args.results_filename, fold), protocol=4)
        valid_log.to_pickle("./logs_{}/valid_log_f{}.pkl".format(args.results_filename, fold), protocol=4)
        torch.save(best_model.state_dict(), './results_{}/best_model_f{}.pt'.format(args.results_filename, fold))
        torch.save(loss_curves, './results_{}/loss_curves_f{}.pt'.format(args.results_filename, fold))
        torch.save(acc_curves, './results_{}/acc_curves_f{}.pt'.format(args.results_filename, fold))


