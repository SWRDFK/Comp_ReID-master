import sys
sys.path.append('../')

import torchvision.transforms as transforms
from .dataset import *
from .loader import *
from tools import *


class Loaders:

    def __init__(self, config):

        self.transform_train = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # dataset
        self.dataset_path = config.dataset_path
        self.train_use = config.train_use

        # batch size
        self.p = config.p
        self.k = config.k

        # dataset paths
        self.samples_path = {
            'comp_train': os.path.join(self.dataset_path, 'train_set/'),
            # test_set A
            # 'comp_test_query': os.path.join(self.dataset_path, 'query_a/'),
            # 'comp_test_gallery': os.path.join(self.dataset_path, 'gallery_a/'),

            # test_set B
            'comp_test_query': os.path.join(self.dataset_path, 'query_b/'),
            'comp_test_gallery': os.path.join(self.dataset_path, 'gallery_b/')}

        # load
        self._load()


    def _load(self):

        # train dataset and iter
        train_samples, self.num_train, self.samples_per_class = self._get_train_samples('comp_train', self.train_use)
        self.train_iter = self._get_uniform_iter(train_samples, self.transform_train, self.p, self.k)

        # test dataset and loader
        self.comp_query_samples, self.comp_gallery_samples = self._get_test_samples('comp_test')
        self.comp_query_loader = self._get_loader(self.comp_query_samples, self.transform_test, 128)
        self.comp_gallery_loader = self._get_loader(self.comp_gallery_samples, self.transform_test, 128)


    def _get_train_samples(self, train_dataset, train_use):

        train_samples_path = self.samples_path[train_dataset]
        samples = Comp_Train_Samples(train_samples_path, train_use)

        return samples, samples.num_train, samples.samples_per_class


    def _get_test_samples(self, test_dataset):

        query_data_path = self.samples_path[test_dataset + '_query']
        gallery_data_path = self.samples_path[test_dataset + '_gallery']

        query_samples = Comp_Test_Samples(query_data_path, reorder=False)
        gallery_samples = Comp_Test_Samples(gallery_data_path, reorder=False)

        return query_samples, gallery_samples


    def _get_uniform_iter(self, samples, transform, p, k):

        dataset = CompDataset(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False,
                                 sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)

        return iters


    def _get_random_iter(self, samples, transform, batch_size):

        dataset = CompDataset(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)

        return iters


    def _get_random_loader(self, samples, transform, batch_size):

        dataset = CompDataset(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader


    def _get_loader(self, samples, transform, batch_size):

        dataset = CompDataset(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

