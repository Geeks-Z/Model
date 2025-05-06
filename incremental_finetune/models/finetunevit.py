import logging
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.inc_net import VITNet
from torch.utils.data.distributed import DistributedSampler as DS

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = VITNet(args, True)
        self.batch_size = args["batch_size"] if args["batch_size"] is not None else 128
        self.init_lr = args["init_lr"] if args["init_lr"] is not None else 0.01
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.train_time = 0
        self.test_time = 0

    def replace_fc(self, trainloader, model, args):
        # 用protonet的方法更新fc层
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            if isinstance(self._network, nn.DataParallel):
                self._network.module.fc.weight.data[class_index] = proto
            else:
                self._network.fc.weight.data[class_index] = proto
        return model

    def train(self, data_manager):
        self._total_classes = data_manager.nb_classes
        self._network = self._network.to(self._device)
        self._network = CustomDistributedDataParallel(self._network, device_ids=[self.args["local_rank"]],
                                                      output_device=self.args["local_rank"])
        self._network.update_fc(self._total_classes)

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(0, self._total_classes),
                                                      source="train", mode="train")
        train_sampler = DS(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       num_workers=num_workers, sampler=train_sampler)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=num_workers,
                                      shuffle=False)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(0, self._total_classes),
                                                              source="train", mode="test")
        train_for_protonet_sampler = DS(train_dataset_for_protonet)
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size,
                                                    num_workers=num_workers, sampler=train_for_protonet_sampler)

        self._train(self.train_loader, self.train_loader_for_protonet)

    def _train(self, train_loader, train_loader_for_protonet):
        self._network.to(self._device)
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                  weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'],
                                                         eta_min=self.min_lr)
        self._init_train(train_loader, optimizer, scheduler)
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device,
                                                                                         non_blocking=True)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
        # total_time = time.time() - start_time
        # self.train_time += round(total_time, 2)
        # print('Training time {}'.format(self.train_time))
        logging.info(info)

    def frozen_params(self):
        frozen_mlp = 'mlps.' + str(self._cur_task)
        for name, p in self._network.named_parameters():
            if frozen_mlp in name:
                p.requires_grad = False

class CustomDistributedDataParallel(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)