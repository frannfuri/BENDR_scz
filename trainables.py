import torch
import time
import copy
import sklearn.metrics as skmetrics
import pandas as pd
from collections import OrderedDict


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()
    train_log = list()
    valid_log = list()

    best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0
    lowest_loss = 99.9

    #train_accs = []
    train_losses = []
    #valid_accs = []
    valid_losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            #running_corrects = 0

            it = 0
            train_num_samples = 0
            val_num_samples = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.to(float)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # TODO: Classification
                    #_, preds = torch.max(outputs[0], 1)
                    loss = criterion(outputs[0].squeeze(), labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        train_metrics = OrderedDict()
                        train_metrics['epoch'] = epoch
                        train_metrics['iter'] = it
                        it += 1
                        train_metrics['loss'] = loss.item()
                        # TODO: REVISAR accuracy !!
                        #train_metrics['accuracy'] = torch.sum(preds == labels.data) / inputs.size(0)
                        #train_metrics['f1score'], train_metrics['preciss'], train_metrics['recall'] = f1_loss(labels,
                        #                                                                                      preds)
                        train_metrics['lr'] = optimizer.param_groups[0]['lr']
                        loss.backward()
                        optimizer.step()
                        train_log.append(train_metrics)
                        train_num_samples += 1 * inputs.size(0)
                    else:
                        val_num_samples += 1 + inputs.size(0)
                        valid_metrics = OrderedDict()
                        valid_metrics['epoch'] = epoch
                        valid_metrics['loss'] = loss.item()
                        #valid_metrics['accuracy'] = torch.sum(preds == labels.data) / inputs.size(0)
                        #valid_metrics['f1score'], valid_metrics['preciss'], valid_metrics['recall'] = f1_loss(labels,
                        #                                                                                      preds)
                        valid_log.append(valid_metrics)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                # PROBAR DE LA OTRA MANERA
                scheduler.step()

            epoch_loss = running_loss  # / dataset_sizes[phase]
            #epoch_acc = running_corrects.double()  # / dataset_sizes[phase]
            if phase == 'train':
                train_losses.append(epoch_loss / train_num_samples)
                #train_accs.append(epoch_acc / train_num_samples)
            else:
                valid_losses.append(epoch_loss / val_num_samples)
                #valid_accs.append(epoch_acc / val_num_samples)

            print('{} Loss: {:.4f} * numb. of samples.'.format(
                phase, epoch_loss))#, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss/val_num_samples
                best_model_wts = copy.deepcopy(model.state_dict())

            #if phase == 'val' and epoch_acc / val_num_samples > best_acc:
            #    best_acc = epoch_acc / val_num_samples
            #    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))
    print('Lowest val Loss: {:4f}'.format(lowest_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, (train_losses, valid_losses), pd.DataFrame(train_log), pd.DataFrame(
        valid_log)


def _simple_accuracy(inputs, outputs):
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    # average over last dimensions
    while len(outputs.shape) >= 3:
        outputs = outputs.mean(dim=-1)
    return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()


def balanced_accuracy(y_t, y_p):
    return skmetrics.balanced_accuracy_score(y_t, y_p)


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1, precision, recall