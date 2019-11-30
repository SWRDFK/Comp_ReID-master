import numpy as np
from PIL import Image
import copy
import os
import random
from tools import os_walk


class Comp_Train_Samples:

    def __init__(self, samples_path, train_use, reorder=True):

        # parameters
        self.samples_path = samples_path
        self.train_use = train_use
        self.reorder = reorder

        # load samples
        samples = self._load_images_path(self.samples_path, self.train_use)

        # reorder person identities
        if self.reorder:
            samples = self._reorder_labels(samples, 1)

        self.samples = samples

        # label_dict of train_set
        self.label_dict = self._make_label_dict("/home/kangning/Competition/train_list.txt")

        # number of training samples
        self.num_train = len(self.samples)

        # number of samples per person identities
        self.samples_per_class = [len(self.label_dict[key]) for key in self.label_dict]


    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        # delete repetitive elements and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples


    def _load_images_path(self, folder_dir, train_use):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)

        # get all train data
        if train_use == 'all':
            for file_name in files_name:
                identi_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identi_id])
            return samples

        # get part train data without one-sample ID
        elif train_use == 'part':
            label_dict = self._make_label_dict("/home/kangning/Competition/train_list.txt")

            for key in label_dict:
                n_pids = len(label_dict[key])
                # for one sample per identity: select 0 sample
                if n_pids == 1:
                    continue
                # for others: select all samples
                else:
                    sel_list = label_dict[key]
                    for file_name in sel_list:
                        identi_id = self._analysis_file_name(file_name)
                        samples.append([root_path + file_name, identi_id])
            return samples


    def _analysis_file_name(self, file_name):
        '''
        :param file_name: label format like train/395017260.png 0
        :return:
        '''
        f = open("/home/kangning/Competition/train_list.txt", "r")
        lines = f.readlines()
        for line in lines:
            if file_name in line:
                identi_id = int(line.split(' ')[1])
                return identi_id


    def _make_label_dict(self, label_path):
        '''
        :param label_path: format like "/home/kangning/Competition/train_list.txt"
        :return: label_dict (key: ID, value: image_name)
        '''
        label_dict = {}

        f = open(label_path, "r")
        lines = f.readlines()
        for line in lines:
            image = line.split(' ')[0].split('/')[1]
            label = line.split(' ')[1].replace('\n', '')
            label_dict.setdefault(label, []).append(image)
        return label_dict



class Comp_Test_Samples:

    def __init__(self, samples_path, reorder=True):

        # parameters
        self.samples_path = samples_path
        self.reorder = reorder

        # load samples
        samples = self._load_images_path(self.samples_path)

        # reorder person identities
        if self.reorder:
            samples = self._reorder_labels(samples, 1)

        self.samples = samples


    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        # delete repetitive elements and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return samples


    def _load_images_path(self, folder_dir):

        samples = []
        root_path, _, files_name = os_walk(folder_dir)

        for file_name in files_name:
            # test_set has no labels, we set id=0 instead.
            identi_id = int(0)
            samples.append([root_path + file_name, identi_id])
        return samples



class CompDataset:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')

