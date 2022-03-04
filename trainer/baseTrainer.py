
import argparse
import datetime
import os
from typing import List, Dict, Tuple

import torch
from torch.nn import DataParallel
from torch.optim import optimizer
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from trainer import util
import tensorboardX

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class BaseTrainer:
    """ Trainer base class with common methods """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._log_paths = dict()
        self._best_results = dict()
        name = str(datetime.datetime.now()).replace(' ', '_').replace(":","_")
        # print(name)
        self._log_path = os.path.join(self.args.log_path, self.args.model_type,self.args.dataset,name)
        self._log_path_result = os.path.join(self._log_path,"result")
        self._log_path_predict = os.path.join(self._log_path, "predict")
        # util.create_directories_dir(self._log_path)
        if hasattr(args, 'save_path'):
            # print(self.args.label+name)
            self._save_path = os.path.join(self.args.save_path, self.args.label+name)
            # print(self._save_path)
            util.create_directories_dir(self._save_path)
            # tensorboard summary
            self._summary_writer = tensorboardX.SummaryWriter(self._log_path) if tensorboardX is not None else None

    def _save_best(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, optimizer: optimizer,
                   accuracy: float, iteration: int, label: str, extra=None):
        if accuracy > self._best_results[label]:
            self._save_model(self._save_path, model, tokenizer, iteration,
                             optimizer=optimizer if self.args.save_optimizer else None,
                             save_as_best=True, name='model_%s' % label, extra=extra)
            self._best_results[label] = accuracy

    def _add_dataset_logging(self, *labels, data: Dict[str, List[str]]):
        for label in labels:
            dic = dict()

            for key, columns in data.items():
                path = os.path.join(self._log_path, '%s_%s.csv' % (key, label))
                # print(self._log_path,path)
                util.create_csv(path, *columns)
                dic[key] = path

            self._log_paths[label] = dic
            self._best_results[label] = 0

    def _save_model(self, save_path: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                    iteration: int, optimizer: optimizer = None, save_as_best: bool = False,
                    extra: dict = None, include_iteration: int = True, name: str = 'model'):
        extra_state = dict(iteration=iteration)

        if optimizer:
            extra_state['optimizer'] = optimizer.state_dict()

        if extra:
            extra_state.update(extra)

        if save_as_best:
            dir_path = os.path.join(save_path, '%s_best' % name)
        else:
            dir_name = '%s_%s' % (name, iteration) if include_iteration else name
            dir_path = os.path.join(save_path, dir_name)

        util.create_directories_dir(dir_path)

        # save model
        if isinstance(model, DataParallel):
            model.module.save_pretrained(dir_path)
        else:
            model.save_pretrained(dir_path)

        # save vocabulary
        tokenizer.save_pretrained(dir_path)

        # save extra
        state_path = os.path.join(dir_path, 'extra.state')
        torch.save(extra_state, state_path)

    def _log_tensorboard(self, dataset_label: str, data_label: str, data: object, iteration: int):
        if self._summary_writer is not None:
            self._summary_writer.add_scalar('data/%s/%s' % (dataset_label, data_label), data, iteration)

    def _log_csv(self, dataset_label: str, data_label: str, *data: Tuple[object]):
        logs = self._log_paths[dataset_label]
        try:
            util.append_csv(logs[data_label], *data)
        except:
            print(dataset_label,logs)

    def _get_lr(self, optimizer):
        lrs = []
        for group in optimizer.param_groups:
            lr_scheduled = group['lr']
            lrs.append(lr_scheduled)
        return lrs

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'test': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'senti_prec_micro', 'senti_rec_micro', 'senti_f1_micro',
                                                 'senti_prec_macro', 'senti_rec_macro', 'senti_f1_macro',
                                                 'senti_nec_prec_micro', 'senti_nec_rec_micro', 'senti_nec_f1_micro',
                                                 'senti_nec_prec_macro', 'senti_nec_rec_macro', 'senti_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  senti_prec_micro: float, senti_rec_micro: float, senti_f1_micro: float,
                  senti_prec_macro: float, senti_rec_macro: float, senti_f1_macro: float,

                  senti_nec_prec_micro: float, senti_nec_rec_micro: float, senti_nec_f1_micro: float,
                  senti_nec_prec_macro: float, senti_nec_rec_macro: float, senti_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):
        # log to csv
        self._log_csv(label, 'test', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      senti_prec_micro, senti_rec_micro, senti_f1_micro,
                      senti_prec_macro, senti_rec_macro, senti_f1_macro,

                      senti_nec_prec_micro, senti_nec_rec_micro, senti_nec_f1_micro,
                      senti_nec_prec_macro, senti_nec_rec_macro, senti_nec_f1_macro,
                      epoch, iteration, global_iteration)
        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/senti_prec_micro', senti_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_recall_micro', senti_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_f1_micro', senti_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_prec_macro', senti_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_recall_macro', senti_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_f1_macro', senti_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/senti_nec_prec_micro', senti_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_nec_recall_micro', senti_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_nec_f1_micro', senti_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_nec_prec_macro', senti_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_nec_recall_macro', senti_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/senti_nec_f1_macro', senti_nec_f1_macro, global_iteration)

