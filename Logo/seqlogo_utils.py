import torch
import torch.nn as nn
import torch.nn.functional as F
import gzip, csv
from utils import dinuc_shuffle, reverse_complement, seq2pad
from Chip import Chip_test, chipseq_dataset
from torch.utils.data import DataLoader

class ConvNet_seq_logo(nn.Module):
    def __init__(self,nummotif,motiflen,poolType,neuType,mode,dropprob, learning_rate,learning_Momentum,sigmaConv,sigmaNeu,beta1,beta2,beta3, device, reverse_complemet_mode):
        super(ConvNet_seq_logo, self).__init__()
        self.poolType=poolType
        self.neuType=neuType
        self.mode=mode
        self.learning_rate=learning_rate
        self.device = device
        self.reverse_complemet_mode=reverse_complemet_mode
        self.momentum_rate=learning_Momentum
        self.sigmaConv=sigmaConv

        self.wConv=torch.randn(nummotif,4,motiflen).to(device)
        torch.nn.init.normal_(self.wConv,mean=0,std=self.sigmaConv)
        self.wConv.requires_grad=True

        self.wRect=torch.randn(nummotif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect=-self.wRect
        self.wRect.requires_grad=True

        self.dropprob=dropprob
        self.sigmaNeu=sigmaNeu
        self.wHidden=torch.randn(2*nummotif,32).to(device)
        self.wHiddenBias=torch.randn(32).to(device)

        if neuType=='nohidden':
            if poolType=='maxavg':
                self.wNeu=torch.randn(2*nummotif,1).to(device)
            else:
                self.wNeu=torch.randn(nummotif,1).to(device)
            self.wNeuBias=torch.randn(1).to(device)
            torch.nn.init.normal_(self.wNeu,mean=0,std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias,mean=0,std=self.sigmaNeu)

        else:
            if poolType=='maxavg':
                self.wHidden=torch.randn(2*nummotif,32).to(device)
            else:
                
                self.wHidden=torch.randn(nummotif,32).to(device)
            self.wNeu=torch.randn(32,1).to(device)
            self.wNeuBias=torch.randn(1).to(device)
            self.wHiddenBias=torch.randn(32).to(device)
            torch.nn.init.normal_(self.wNeu,mean=0,std=self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias,mean=0,std=self.sigmaNeu)
            torch.nn.init.normal_(self.wHidden,mean=0,std=0.3)
            torch.nn.init.normal_(self.wHiddenBias,mean=0,std=0.3)
            
  
            self.wHidden.requires_grad=True
            self.wHiddenBias.requires_grad=True
            #wHiddenBias=tf.Variable(tf.truncated_normal([32,1],mean=0,stddev=sigmaNeu)) #hidden bias for everything

        self.wNeu.requires_grad=True
        self.wNeuBias.requires_grad=True

        self.beta1=beta1
        self.beta2=beta2
        self.beta3=beta3
    
    def divide_two_tensors(self,x):
        l=torch.unbind(x)
        list1=[l[2*i] for i in range(int(x.shape[0]/2))]
        list2=[l[2*i+1] for i in range(int(x.shape[0]/2))]
        x1=torch.stack(list1,0)
        x2=torch.stack(list2,0)
        return x1,x2

    def forward_pass(self,x,mask=None,use_mask=False):
        conv=F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect=conv.clamp(min=0)

        return rect
        
    def forward(self, x):
        
        if not  self.reverse_complemet_mode:
            out= self.forward_pass(x)
        else:
            print("not supported error")

        return out

def sequence_loader(path, reverse_mode):
    seq = []

    with gzip.open(path, 'rt') as data:
        next(data)
        reader = csv.reader(data,delimiter='\t')
        if not reverse_mode:
            for row in reader:
                    seq.append([row[2],[int(row[3])]])
        else:
            for row in reader:
                    seq.append([row[2],[int(row[3])]])
                    seq.append([reverse_complement(row[2]),[int(row[3])]])
    
    return seq


# Deep Bind Model
def retification_results(name, dataset, device, reverse_mode):
    checkpoint = torch.load('./Models/'+name+'.pth')

    best_hyperparameters = torch.load('./Hyperparameters/'+name+'.pth')

    best_poolType=best_hyperparameters['best_poolType']
    best_neuType=best_hyperparameters['best_neuType']
    best_learning_steps=best_hyperparameters['best_learning_steps']
    best_lr=best_hyperparameters['best_LearningRate']
    best_droprate=best_hyperparameters['best_dropprob']
    best_momentum=best_hyperparameters['best_LearningMomentum']
    best_sigmaConv=best_hyperparameters['best_sigmaConv']
    best_sigmaNeu=best_hyperparameters['best_sigmaNeu']
    best_beta1=best_hyperparameters['best_beta1']
    best_beta2=best_hyperparameters['best_beta2']
    best_beta3=best_hyperparameters['best_beta3']

    model = ConvNet_seq_logo(16, 24, best_poolType, best_neuType, 'test', best_lr, best_momentum, best_sigmaConv, best_droprate, best_sigmaNeu, best_beta1, best_beta2, best_beta3, device, reverse_mode).to(device)
    model.wConv=checkpoint['conv']
    model.wRect=checkpoint['rect']
    model.wHidden=checkpoint['wHidden']
    model.wHiddenBias=checkpoint['wHiddenBias']
    model.wNeu=checkpoint['wNeu']
    model.wNeuBias=checkpoint['wNeuBias']

    seq = sequence_loader(dataset, reverse_mode)

    seq_data = []

    for l in range(len(seq)):
        seq_data.extend([[seq2pad(seq[l][0], motiflen=24), seq[l][1]]])
    
    seq_dataset=chipseq_dataset(seq_data)
    batchSize=seq_dataset.__len__()
    test_loader = DataLoader(dataset=seq_dataset,batch_size=batchSize,shuffle=False)

    with torch.no_grad():
        model.mode = 'test'
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
    
    return seq, seq_data, output