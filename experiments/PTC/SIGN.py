import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch

from model.GC_net import SIGN_net
from impl.binGraphClassifier import modelImplementation_GraphBinClassifier
from utils.utils_methods import printParOnFile
from data_reader.cross_validation_reader import getcross_validation_split

if __name__ == '__main__':

    n_epochs = 400
    n_classes = 2
    dataset_path = '~/Dataset/'
    dataset_name = 'PTC_MR'
    n_folds = 10
    test_epoch = 1
    n_run = 5

    #GRID parameter sets
    n_unit_list = [15, 30, 60]
    lr_list = [1e-3, 5e-4, 1e-4]
    weight_decay_list = [5e-3, 5e-4]
    drop_prob_list = [0.4,0.6]
    batch_size_list = [16, 32]
    max_k_list=[3,4,5,6]
    output_list = ["funnel", "restricted_funnel"]
    n_layers_list=[1]
    for run in range(n_run):
        for n_units in n_unit_list:
            for lr in lr_list:
                for drop_prob in drop_prob_list:
                    for weight_decay in weight_decay_list:
                        for batch_size in batch_size_list:
                            for output in output_list:
                                for n_layers in n_layers_list:
                                    for max_k in max_k_list:

                                        test_type = "SIGN_GNN"

                                        test_name = "run-"+str(run)+"_"+test_type + "_data-" + dataset_name + "_nFold-" + str(
                                            n_folds) + "_lr-" + \
                                                    str(lr) + "_drop_prob-" + str(drop_prob) + "_weight-decay-" + str(
                                            weight_decay) + \
                                                    "_batchSize-" + str(batch_size) + "_nHidden-" + str(
                                            n_units) + "_output-" + \
                                                    str(output) + "_maxK-" + str(max_k) + "_n_layers-" + str(n_layers)
                                        training_log_dir = os.path.join("./test_log/" + test_type, test_name)
                                        if not os.path.exists(training_log_dir):
                                            os.makedirs(training_log_dir)

                                            printParOnFile(test_name=test_name, log_dir=training_log_dir,
                                                           par_list={"dataset_name": dataset_name,
                                                                     "n_fold": n_folds,
                                                                     "learning_rate": lr,
                                                                     "drop_prob": drop_prob,
                                                                     "weight_decay": weight_decay,
                                                                     "batch_size": batch_size,
                                                                     "n_hidden": n_units,
                                                                     "test_epoch": test_epoch,
                                                                     "output": output,
                                                                     "max_k": max_k,
                                                                     "n_layers":n_layers})

                                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                                            criterion = torch.nn.NLLLoss()

                                            dataset_cv_splits = getcross_validation_split(dataset_path, dataset_name, n_folds,
                                                                                          batch_size,K=max_k)
                                            for split_id, split in enumerate(dataset_cv_splits):
                                                loader_train = split[0]
                                                loader_test = split[1]
                                                loader_valid = split[2]

                                                model = SIGN_net(in_channels=loader_train.dataset.num_features,
                                                                    n_gc_hidden_units=n_units,
                                                                    k=max_k,
                                                                    n_layer=n_layers,
                                                                    n_class=n_classes,
                                                                    drop_prob=drop_prob,
                                                                    output=output
                                                                    )

                                                model_impl = modelImplementation_GraphBinClassifier(model, lr, criterion,
                                                                                                    device).to(device)

                                                model_impl.set_optimizer(weight_decay=weight_decay)

                                                model_impl.train_test_model(split_id, loader_train, loader_test, loader_valid,
                                                                            n_epochs, test_epoch, test_name, training_log_dir)
                                                if str(device) == 'cuda':
                                                    del model
                                                    del model_impl
                                                    torch.cuda.empty_cache()
                                        else:
                                            print("test has been already execute")

