import os
import torch
import torchvision
import torchvision.transforms as transforms

class Imagenet:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        traindir = os.path.join(self.config.data_path, 'train')
        valdir = os.path.join(self.config.data_path, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        self.data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.train_batch_size,
            num_workers = self.config.num_workers,
            pin_memory = self.config.pin_memory,
            sampler=train_sampler
            )

        self.data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=self.config.test_batch_size,
            num_workers = self.config.num_workers,
            pin_memory = self.config.pin_memory,
            sampler=test_sampler
            )
