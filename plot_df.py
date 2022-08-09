import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    f1score_available = False
    final_file_name = 'decomp_study_lr0.0001bs16_deltaPANSS_posit'
    metric = 'MSE  (' + r'$\Delta$ ' + 'PANSS)'
    title_caption = '[All. subjs.]'
    path = './logs_new/'
    ################################3
    # Curves my implementation #2 (Dataframe)
    #path = './logs_new_mae/'
    a0 = pd.read_pickle(path + 'train_log_f0_{}.pkl'.format(final_file_name))
    a1 = pd.read_pickle(path + 'train_log_f1_{}.pkl'.format(final_file_name))
    a2 = pd.read_pickle(path + 'train_log_f2_{}.pkl'.format(final_file_name))
    a3 = pd.read_pickle(path + 'train_log_f3_{}.pkl'.format(final_file_name))
    n_epochs = len(a0['epoch'].unique())

    b0 = pd.read_pickle(path + 'valid_log_f0_{}.pkl'.format(final_file_name))
    b1 = pd.read_pickle(path + 'valid_log_f1_{}.pkl'.format(final_file_name))
    b2 = pd.read_pickle(path + 'valid_log_f2_{}.pkl'.format(final_file_name))
    b3 = pd.read_pickle(path + 'valid_log_f3_{}.pkl'.format(final_file_name))

    tr_loss_f0_ = []
    val_loss_f0_ = []
    for i in range(n_epochs):
        tr_loss_f0_.append(a0[a0['epoch'] == (i)].mean()['loss'])
        val_loss_f0_.append(b0[b0['epoch'] == (i)].mean()['loss'])
    tr_loss_f1_ = []
    val_loss_f1_ = []
    for i in range(n_epochs):
        tr_loss_f1_.append(a1[a1['epoch'] == (i)].mean()['loss'])
        val_loss_f1_.append(b1[b1['epoch'] == (i)].mean()['loss'])
    tr_loss_f2_ = []
    val_loss_f2_ = []
    for i in range(n_epochs):
        tr_loss_f2_.append(a2[a2['epoch'] == (i)].mean()['loss'])
        val_loss_f2_.append(b2[b2['epoch'] == (i)].mean()['loss'])
    tr_loss_f3_ = []
    val_loss_f3_ = []
    for i in range(n_epochs):
        tr_loss_f3_.append(a3[a3['epoch'] == (i)].mean()['loss'])
        val_loss_f3_.append(b3[b3['epoch'] == (i)].mean()['loss'])

    all_train_mean_loss_ = np.mean(np.array([tr_loss_f0_, tr_loss_f1_, tr_loss_f2_, tr_loss_f3_]), axis=0)
    all_val_mean_loss_ = np.mean(np.array([val_loss_f0_, val_loss_f1_, val_loss_f2_, val_loss_f3_]), axis=0)

    plt.figure(figsize=(6,5))
    plt.plot(all_train_mean_loss_, label='Mean train', lw=2)
    plt.plot(all_val_mean_loss_, label='Mean val', lw=2)
    plt.title('Training loss (mean of the fold-iterations)\n{}'.format(title_caption))
    plt.grid()
    plt.plot(tr_loss_f0_, label='train f1', ls='dotted')
    plt.plot(val_loss_f0_, label='val f1', ls='dotted')
    plt.plot(tr_loss_f1_, label='train f2', ls='dotted')
    plt.plot(val_loss_f1_, label='val f2', ls='dotted')
    plt.plot(tr_loss_f2_, label='train f3', ls='dotted')
    plt.plot(val_loss_f2_, label='val f3', ls='dotted')
    plt.plot(tr_loss_f3_, label='train f4', ls='dotted')
    plt.plot(val_loss_f3_, label='val f4', ls='dotted')
    plt.legend(loc='best', prop={'size': 8})
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    # plt.ylim([0.69, 0.72])
    plt.show()
    a = 0

'''
    tr_loss_f0_ = []
    val_loss_f0_ = []
    for i in range(n_epochs):
        tr_loss_f0_.append(a0[a0['epoch'] == (i)].mean()['accuracy'])
        val_loss_f0_.append(b0[b0['epoch'] == (i)].mean()['accuracy'])
    tr_loss_f1_ = []
    val_loss_f1_ = []
    for i in range(n_epochs):
        tr_loss_f1_.append(a1[a1['epoch'] == (i)].mean()['accuracy'])
        val_loss_f1_.append(b1[b1['epoch'] == (i)].mean()['accuracy'])
    tr_loss_f2_ = []
    val_loss_f2_ = []
    for i in range(n_epochs):
        tr_loss_f2_.append(a2[a2['epoch'] == (i)].mean()['accuracy'])
        val_loss_f2_.append(b2[b2['epoch'] == (i)].mean()['accuracy'])
    tr_loss_f3_ = []
    val_loss_f3_ = []
    for i in range(n_epochs):
        tr_loss_f3_.append(a3[a3['epoch'] == (i)].mean()['accuracy'])
        val_loss_f3_.append(b3[b3['epoch'] == (i)].mean()['accuracy'])

    all_train_mean_loss_ = np.mean(np.array([tr_loss_f0_, tr_loss_f1_, tr_loss_f2_, tr_loss_f3_]), axis=0)
    all_val_mean_loss_ = np.mean(np.array([val_loss_f0_, val_loss_f1_, val_loss_f2_, val_loss_f3_]), axis=0)

    plt.figure()
    plt.plot(all_train_mean_loss_, label='Mean train', lw=2)
    plt.plot(all_val_mean_loss_, label='Mean val', lw=2)
    plt.title('Training accuracy (mean of the fold-iterations)')
    plt.plot(tr_loss_f0_, label='train f1', ls='dotted')
    plt.plot(val_loss_f0_, label='val f1', ls='dotted')
    plt.plot(tr_loss_f1_, label='train f2', ls='dotted')
    plt.plot(val_loss_f1_, label='val f2', ls='dotted')
    plt.plot(tr_loss_f2_, label='train f3', ls='dotted')
    plt.plot(val_loss_f2_, label='val f3', ls='dotted')
    plt.plot(tr_loss_f3_, label='train f4', ls='dotted')
    plt.plot(val_loss_f3_, label='val f4', ls='dotted')
    plt.grid()
    # plt.ylim([0.3, 0.8])
    plt.legend(loc='best')

    if f1score_available:
        tr_loss_f0_ = []
        val_loss_f0_ = []
        for i in range(n_epochs):
            tr_loss_f0_.append(a0[a0['epoch'] == (i)].mean()['f1score'])
            val_loss_f0_.append(b0[b0['epoch'] == (i)].mean()['f1score'])
        tr_loss_f1_ = []
        val_loss_f1_ = []
        for i in range(n_epochs):
            tr_loss_f1_.append(a1[a1['epoch'] == (i)].mean()['f1score'])
            val_loss_f1_.append(b1[b1['epoch'] == (i)].mean()['f1score'])
        tr_loss_f2_ = []
        val_loss_f2_ = []
        for i in range(n_epochs):
            tr_loss_f2_.append(a2[a2['epoch'] == (i)].mean()['f1score'])
            val_loss_f2_.append(b2[b2['epoch'] == (i)].mean()['f1score'])
        tr_loss_f3_ = []
        val_loss_f3_ = []
        for i in range(n_epochs):
            tr_loss_f3_.append(a3[a3['epoch'] == (i)].mean()['f1score'])
            val_loss_f3_.append(b3[b3['epoch'] == (i)].mean()['f1score'])

        all_train_mean_loss_ = np.mean(np.array([tr_loss_f0_, tr_loss_f1_, tr_loss_f2_, tr_loss_f3_]), axis=0)
        all_val_mean_loss_ = np.mean(np.array([val_loss_f0_, val_loss_f1_, val_loss_f2_, val_loss_f3_]), axis=0)

        plt.figure()
        plt.plot(all_train_mean_loss_, label='F1 score train', lw=2)
        plt.grid()
        plt.plot(all_val_mean_loss_, label='F1 score val', lw=2)
        plt.title('Training F1score (mean of the fold-iterations)')
        plt.legend(loc='best')

        # plt.ylim([0.0, 1.00])

        tr_loss_f0_ = []
        val_loss_f0_ = []
        for i in range(n_epochs):
            tr_loss_f0_.append(a0[a0['epoch'] == (i)].mean()['preciss'])
            val_loss_f0_.append(b0[b0['epoch'] == (i)].mean()['preciss'])
        tr_loss_f1_ = []
        val_loss_f1_ = []
        for i in range(n_epochs):
            tr_loss_f1_.append(a1[a1['epoch'] == (i)].mean()['preciss'])
            val_loss_f1_.append(b1[b1['epoch'] == (i)].mean()['preciss'])
        tr_loss_f2_ = []
        val_loss_f2_ = []
        for i in range(n_epochs):
            tr_loss_f2_.append(a2[a2['epoch'] == (i)].mean()['preciss'])
            val_loss_f2_.append(b2[b2['epoch'] == (i)].mean()['preciss'])
        tr_loss_f3_ = []
        val_loss_f3_ = []
        for i in range(n_epochs):
            tr_loss_f3_.append(a3[a3['epoch'] == (i)].mean()['preciss'])
            val_loss_f3_.append(b3[b3['epoch'] == (i)].mean()['preciss'])

        all_train_mean_loss_ = np.mean(np.array([tr_loss_f0_, tr_loss_f1_, tr_loss_f2_, tr_loss_f3_]), axis=0)
        all_val_mean_loss_ = np.mean(np.array([val_loss_f0_, val_loss_f1_, val_loss_f2_, val_loss_f3_]), axis=0)

        plt.plot(all_train_mean_loss_, label='Precission train', lw=1)
        plt.plot(all_val_mean_loss_, label='Precission val', lw=1)
        plt.grid()
        plt.legend(loc='best')

        tr_loss_f0_ = []
        val_loss_f0_ = []
        for i in range(n_epochs):
            tr_loss_f0_.append(a0[a0['epoch'] == (i)].mean()['recall'])
            val_loss_f0_.append(b0[b0['epoch'] == (i)].mean()['recall'])
        tr_loss_f1_ = []
        val_loss_f1_ = []
        for i in range(n_epochs):
            tr_loss_f1_.append(a1[a1['epoch'] == (i)].mean()['recall'])
            val_loss_f1_.append(b1[b1['epoch'] == (i)].mean()['recall'])
        tr_loss_f2_ = []
        val_loss_f2_ = []
        for i in range(n_epochs):
            tr_loss_f2_.append(a2[a2['epoch'] == (i)].mean()['recall'])
            val_loss_f2_.append(b2[b2['epoch'] == (i)].mean()['recall'])
        tr_loss_f3_ = []
        val_loss_f3_ = []
        for i in range(n_epochs):
            tr_loss_f3_.append(a3[a3['epoch'] == (i)].mean()['recall'])
            val_loss_f3_.append(b3[b3['epoch'] == (i)].mean()['recall'])

        all_train_mean_loss_ = np.mean(np.array([tr_loss_f0_, tr_loss_f1_, tr_loss_f2_, tr_loss_f3_]), axis=0)
        all_val_mean_loss_ = np.mean(np.array([val_loss_f0_, val_loss_f1_, val_loss_f2_, val_loss_f3_]), axis=0)

        plt.plot(all_train_mean_loss_, label='Recall train', lw=1)
        plt.plot(all_val_mean_loss_, label='Recall val', lw=1)
        plt.grid()
        plt.legend(loc='best')
'''