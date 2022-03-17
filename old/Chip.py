import csv
import gzip
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from utils import seq2pad, dinuc_shuffle, complement, reverse_complement, datasets, logsampler, sqrtsampler

class Chip():
    '''
    Chip class takes a filename as an input\n
    openFile() returns the corresponding train,valid sets
    '''
    def __init__(self, filename, motiflen=24, reverse_complement_mode = False, ):
        self.file = filename
        self.motiflen = motiflen
        self.reverse_complement_mode = reverse_complement_mode
    
    def openFile(self):
        train_dataset = []
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data, delimiter = '\t')
            if not self.reverse_complement_mode:
                for row in reader:
                    train_dataset.append([seq2pad(row[2], self.motiflen), [1]]) # target = 1
                    train_dataset.append([seq2pad(dinuc_shuffle(row[2]), self.motiflen), [0]]) # target = 0
            else:
                for row in reader:
                      train_dataset.append([seq2pad(row[2],self.motiflen),[1]])
                      train_dataset.append([seq2pad(reverse_complement(row[2]),self.motiflen),[1]])
                      train_dataset.append([seq2pad(dinuc_shuffle(row[2]),self.motiflen),[0]])
                      train_dataset.append([seq2pad(dinuc_shuffle(reverse_complement(row[2])),self.motiflen),[0]])
        
        train_dataset_pad = train_dataset

        size = int(len(train_dataset_pad)/3)
        firstvalid = train_dataset_pad[:size]
        secondvalid = train_dataset_pad[size:2*size]
        thirdvalid = train_dataset_pad[2*size:]
        firsttrain = secondvalid+thirdvalid
        secondtrain = firstvalid+thirdvalid
        thirdtrain = firstvalid+secondvalid

        return firsttrain, firstvalid, secondtrain, secondvalid, thirdtrain, thirdvalid, train_dataset_pad

class Chip_test():
    def __init__(self,filename,motiflen=24,reverse_complemet_mode=False):
        self.file = filename
        self.motiflen = motiflen
        self.reverse_complemet_mode=reverse_complemet_mode
            
    def openFile(self):
        test_dataset=[]
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data,delimiter='\t')
            if not self.reverse_complemet_mode:
              for row in reader:
                      test_dataset.append([seq2pad(row[2],self.motiflen),[int(row[3])]])
            else:
              for row in reader:
                      test_dataset.append([seq2pad(row[2],self.motiflen),[int(row[3])]])
                      test_dataset.append([seq2pad(reverse_complement(row[2]),self.motiflen),[int(row[3])]])
            
        return test_dataset

class chipseq_dataset(Dataset):
    '''
    To use DataLodaer() function, This class defines __len__ & __getitem__
    '''
    def __init__(self, xy = None):
        self.x_data = np.asarray([element[0] for element in xy], dtype=np.float32)
        self.y_data = np.asarray([element[1] for element in xy], dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.len = len(self.x_data)
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

def data_loader(train1, valid1, train2, valid2, train3, valid3, all_data, batch_size, reverse_mode):
    '''
    returns train, valid, and all dataloader
    '''
    train_data_1=chipseq_dataset(train1)
    train_data_2=chipseq_dataset(train2)
    train_data_3=chipseq_dataset(train3)
    valid_data_1=chipseq_dataset(valid1)
    valid_data_2=chipseq_dataset(valid2)
    valid_data_3=chipseq_dataset(valid3)
    all_data    =chipseq_dataset(all_data)

    if reverse_mode:
        train_loader1 = DataLoader(dataset=train_data_1,batch_size=batch_size,shuffle=False)
        train_loader2 = DataLoader(dataset=train_data_2,batch_size=batch_size,shuffle=False)
        train_loader3 = DataLoader(dataset=train_data_3,batch_size=batch_size,shuffle=False)
        valid_loader1 = DataLoader(dataset=valid_data_1,batch_size=batch_size,shuffle=False)
        valid_loader2 = DataLoader(dataset=valid_data_2,batch_size=batch_size,shuffle=False)
        valid_loader3 = DataLoader(dataset=valid_data_3,batch_size=batch_size,shuffle=False)
        all_data_loader=DataLoader(dataset=all_data,batch_size=batch_size,shuffle=False)
    else:
        train_loader1 = DataLoader(dataset=train_data_1,batch_size=batch_size,shuffle=True)
        train_loader2 = DataLoader(dataset=train_data_2,batch_size=batch_size,shuffle=True)
        train_loader3 = DataLoader(dataset=train_data_3,batch_size=batch_size,shuffle=True)
        valid_loader1 = DataLoader(dataset=valid_data_1,batch_size=batch_size,shuffle=False)
        valid_loader2 = DataLoader(dataset=valid_data_2,batch_size=batch_size,shuffle=False)
        valid_loader3 = DataLoader(dataset=valid_data_3,batch_size=batch_size,shuffle=False)
        all_data_loader=DataLoader(dataset=all_data,batch_size=batch_size,shuffle=False)
    
    train_data_lodaer = [train_loader1, train_loader2, train_loader3]
    valid_data_loader = [valid_loader1, valid_loader2, valid_loader3]
    all_data_loader = all_data_loader
    
    return train_data_lodaer, valid_data_loader, all_data_loader

    