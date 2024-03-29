## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##
import sys
import copy
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
import numpy as np
from datasets import loaders
from model_defs import Flatten, model_mlp_any, model_cnn_1layer, model_cnn_2layer, model_cnn_4layer, model_cnn_3layer, so2sat_model_cnn_2layer
from bound_layers import BoundSequential
import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime


from config import load_config, get_path, config_modelloader, config_dataloader
from argparser import argparser
from train import Train, Logger
# sys.settrace(gpu_profile)


def main(args):
    config = load_config(args)
    global_eval_config = config["eval_params"]

    models, model_names = config_modelloader(config, load_pretrain = True)
    converted_models = [BoundSequential.convert(model) for model in models]

    robust_errs = []
    errs = []
    for model, model_id, model_config in zip(converted_models, model_names, config["models"]):
        model = model.cuda()

        # make a copy of global training config, and update per-model config
        eval_config = copy.deepcopy(global_eval_config)
        if "eval_params" in model_config:
            eval_config.update(model_config["eval_params"])

        # read training parameters from config file
        method = eval_config["method"]
        verbose = eval_config["verbose"]
        eps = eval_config["epsilon"]
        # parameters specific to a training method
        method_param = eval_config["method_params"]
        norm = float(eval_config["norm"])
        train_data, test_data = config_dataloader(config, **eval_config["loader_params"])

        model_name = get_path(config, model_id, "model", load = False)
        print(model_name)
        model_log = get_path(config, model_id, "eval_log")
        logger = Logger(open(model_log, "w"))
        logger.log("evaluation configurations:", eval_config)
            
        logger.log("Evaluating...")
        # evaluate

        import pdb
        pdb.set_trace()

        robust_err, err = Train(model, model_id, 0, test_data, eps, eps, eps, norm, logger, verbose, False, None, method, **method_param)
        robust_errs.append(robust_err)
        errs.append(err)

    print('model robust errors (for robustly trained models, not valid for naturally trained models):')
    print(robust_errs)
    robust_errs = np.array(robust_errs)
    print('min: {:.4f}, max: {:.4f}, median: {:.4f}, mean: {:.4f}'.format(np.min(robust_errs), np.max(robust_errs), np.median(robust_errs), np.mean(robust_errs)))
    print('clean errors for models with min, max and median robust errors')
    i_min = np.argmin(robust_errs)
    i_max = np.argmax(robust_errs)
    i_median = np.argsort(robust_errs)[len(robust_errs) // 2]
    print('for min: {:.4f}, for max: {:.4f}, for median: {:.4f}'.format(errs[i_min], errs[i_max], errs[i_median]))
    print('model clean errors:')
    print(errs)
    print('min: {:.4f}, max: {:.4f}, median: {:.4f}, mean: {:.4f}'.format(np.min(errs), np.max(errs), np.median(errs), np.mean(errs)))


if __name__ == "__main__":
    args = argparser()
    main(args)
