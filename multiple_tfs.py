import os
import torch
import numpy as np

import math 
import random

from os import path

import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader

from utils import seq2pad, dinuc_shuffle, complement, reverse_complement, datasets, logsampler, sqrtsampler
from Chip import Chip, chipseq_dataset, data_loader, Chip_test
from network import ConvNet, ConvNet_test

print(torch.__version__)
print('using gpu : ', torch.cuda.is_available())

# Settings
num_motif = 16
bases = 'ACGT' # DNA bases
# basesRNA = 'ACGU' # RNA bases
dictReverse = {'A':'T','C':'G','G':'C','T':'A','N':'N'} #dictionary to implement reverse-complement mode
reverse_mode=False
# Device Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
epochs = 5
num_classes = 10
batch_size = 100
lr = 0.001
# dataset
path = './data/encode/'
dataset_names = datasets(path)
num_tf = len(dataset_names[0])
num_tf = 2

for i in range(num_tf):
    name = dataset_names[0][i]
    name = name.split(path)[1].split("_AC")[0]

    print("TF number ", i+1, " : ", name)
    
    chipseq = Chip(dataset_names[0][i]) # './data/encode/ELK1_GM12878_ELK1_(1277-1)_Stanford_AC.seq.gz'

    train1, valid1, train2, valid2, train3, valid3, all_data = chipseq.openFile()
    train_data_loader, valid_data_loader, all_data_loader = data_loader(train1, valid1, train2, valid2, train3, valid3, all_data, batch_size, reverse_mode)

    # Hyperparameter Learning

    AUC_best = 0
    learning_steps_list = [4000, 8000, 12000, 16000, 20000]

    for epoch in range(epochs):
        pool_list = ['max', 'maxavg']
        random_pool = random.choice(pool_list)

        neuType_list = ['hidden', 'nohidden']
        random_neuType = random.choice(neuType_list)

        dropout_list = [0.5, 0.75, 1.0]
        drop_rate = random.choice(dropout_list)

        lr = logsampler(0.0005, 0.05)
        momentum_rate = sqrtsampler(0.95, 0.99)
        sigmaConv = logsampler(10**-7, 10**-3)
        sigmaNeu=logsampler(10**-5,10**-2) 
        beta1=logsampler(10**-15,10**-3)
        beta2=logsampler(10**-10,10**-3)
        beta3=logsampler(10**-10,10**-3)

        model_AUC = [[], [], []]

        for idx in range(3):
            model  = ConvNet(16, 24, random_pool, random_neuType, 'training', drop_rate, lr, momentum_rate, sigmaConv, sigmaNeu, beta1, beta2, beta3, device, reverse_complement_mode=reverse_mode).to(device)
            if random_neuType == 'nohidden':
                optimizer = torch.optim.SGD([model.wConv, model.wRect, model.wNeu, model.wNeuBias], lr = model.lr, momentum=model.momentum_rate, nesterov=True)
            else:
                optimizer = torch.optim.SGD([model.wConv, model.wRect, model.wNeu, model.wNeuBias, model.wHidden, model.wHiddenBias], lr = model.lr, momentum=model.momentum_rate, nesterov=True)
            
            train_loader = train_data_loader[idx]
            valid_loader = valid_data_loader[idx]

            learning_steps = 0
            while learning_steps <= 20000:
                model.mode = 'training'
                auc = []
                for i, (data, target) in enumerate(train_loader):
                    data = data.to(device)
                    target = target.to(device)
                    if model.reverse_complement_mode:
                        target_2 = torch.randn(int(target.shape[0]/2), 1) # 뭐하는 부분인지 이해 안됨!! -> target.shape[0] = 64
                        for i in range(target_2.shape[0]):
                            target_2[i] = target[2*i]
                        target = target_2.to(device)
                    
                    # Forward Pass
                    output = model(data)
                    if model.neuType == 'nohidden':
                        loss = F.binary_cross_entropy(torch.sigmoid(output), target) + model.beta1*model.wConv.norm() + model.beta3*model.wNeu.norm()
                    else: 
                        loss = F.binary_cross_entropy(torch.sigmoid(output), target) + model.beta1*model.wConv.norm() + model.beta2*model.wHidden.norm() + model.beta3*model.wNeu.norm()
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    learning_steps+=1

                    if learning_steps%4000 == 0:
                        
                        with torch.no_grad():
                            model.mode = 'test'
                            auc = []
                            for i, (data, target) in enumerate(valid_loader):
                                data = data.to(device)
                                target = target.to(device)
                                if model.reverse_complement_mode:
                                    target_2 = torch.randn(int(target.shape[0]/2), 1)
                                    for i in range(target_2.shape[0]):
                                        target_2[i] = target[2*i]
                                    target = target_2.to(device)
                                # Forward Pass
                                output = model(data)
                                pred_sig = torch.sigmoid(output)
                                pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                                labels = target.cpu().numpy().reshape(output.shape[0])

                                auc.append(metrics.roc_auc_score(labels, pred))

                            model_AUC[idx].append(np.mean(auc))
                            print("Epoch", epoch+1, " with Training Fold ", idx+1, " & learning steps ", learning_steps_list[len(model_AUC[idx])-1], " - AUC : ", np.mean(auc))
        
        print('---'*5)
        for n in range(5):
            AUC = (model_AUC[0][n] + model_AUC[1][n] + model_AUC[2][n])/3
            if (AUC > AUC_best):
                AUC_best = AUC
                best_learning_steps = learning_steps_list[n]
                best_lr = model.lr
                best_momentum = model.momentum_rate
                best_neuType = model.neuType
                best_poolType = model.poolType
                best_sigmaConv = model.sigmaConv
                best_droprate = model.droprate
                best_sigmaNeu = model.sigmaNeu
                best_beta1 = model.beta1
                best_beta2 = model.beta2
                best_beta3 = model.beta3

    print('best_poolType=',best_poolType)
    print('best_neuType=',best_neuType)
    print('best_AUC=',AUC_best)
    print('best_learning_steps=',best_learning_steps)      
    print('best_LearningRate=',best_lr)
    print('best_LearningMomentum=',best_momentum)
    print('best_sigmaConv=',best_sigmaConv)
    print('best_dropprob=',best_droprate)
    print('best_sigmaNeu=',best_sigmaNeu)
    print('best_beta1=',best_beta1)
    print('best_beta2=',best_beta2)
    print('best_beta3=',best_beta3)

    best_hyperparameters = {'best_poolType': best_poolType,'best_neuType':best_neuType,'best_learning_steps':best_learning_steps,'best_LearningRate':best_lr,
                            'best_LearningMomentum':best_momentum,'best_sigmaConv':best_sigmaConv,'best_dropprob':best_droprate,
                            'best_sigmaNeu':best_sigmaNeu,'best_beta1':best_beta1, 'best_beta2':best_beta2,'best_beta3':best_beta3}

    # Save Hyperparameters
    torch.save(best_hyperparameters, './Hyperparameters/'+name+'.pth')

    # Model Training

    AUC_best = 0
    learning_steps_list=[4000,8000,12000,16000,20000]

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

    for number_models in range(6):
        model = ConvNet_test(16, 24, best_poolType, best_neuType, 'training', best_lr, best_momentum, best_sigmaConv, best_droprate, best_sigmaNeu, best_beta1, best_beta2, best_beta3, device, False).to(device)

        if model.neuType == 'nohidden':
            optimizer = torch.optim.SGD([model.wConv, model.wRect, model.wNeu, model.wNeuBias], lr = model.learning_rate, momentum= model.momentum_rate, nesterov=True)
        else:
            optimizer = torch.optim.SGD([model.wConv, model.wRect, model.wNeu, model.wNeuBias, model.wHidden, model.wHiddenBias], lr = model.learning_rate, momentum=model.momentum_rate, nesterov=True)
        
        train_loader = all_data_loader
        valid_loader = all_data_loader
        learning_steps = 0

        while learning_steps <= best_learning_steps:
            for i, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                if reverse_mode:
                    target_2 = torch.randn(int(target.shape[0]/2), 1)
                    for i in range(target_2.shape[0]):
                        target_2[i] = target[2*i]
                    target = target_2.to(device)
                
                # Forward Pass
                output = model(data)
                if model.neuType == 'nohidden':
                    loss = F.binary_cross_entropy(torch.sigmoid(output), target) + model.beta1*model.wConv.norm() + model.beta3*model.wNeu.norm()
                else: 
                    loss = F.binary_cross_entropy(torch.sigmoid(output), target) + model.beta1*model.wConv.norm() + model.beta2*model.wHidden.norm() + model.beta3*model.wNeu.norm()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learning_steps += 1
        
        with torch.no_grad():
            model.mode = 'test'
            auc = []
            for i, (data, target) in enumerate(valid_loader):
                data = data.to(device)
                target = target.to(device)
                if reverse_mode:
                    target_2 = torch.randn(int(target.shape[0]/2), 1)
                    for i in range(target_2.shape[0]):
                        target_2[i] = target[2*i]
                    target = target_2.to(device)
                
                # Forward Pass
                output = model(data)
                pred_sig = torch.sigmoid(output)
                pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                labels = target.cpu().numpy().reshape(output.shape[0])

                auc.append(metrics.roc_auc_score(labels, pred))
            
            AUC_training = np.mean(auc)
            print('AUC for model ', number_models, ' = ', AUC_training)
            if AUC_training > AUC_best:
                state = {'conv': model.wConv,'rect':model.wRect,'wHidden':model.wHidden,'wHiddenBias':model.wHiddenBias,'wNeu':model.wNeu,'wNeuBias':model.wNeuBias}
                # Save Models
                torch.save(state, './Models/'+name+'.pth')

    checkpoint = torch.load('./Models/'+name+'.pth')
    model = ConvNet_test(16, 24, best_poolType, best_neuType, 'test', best_lr, best_momentum, best_sigmaConv, best_droprate, best_sigmaNeu, best_beta1, best_beta2, best_beta3, device, reverse_mode).to(device)
    model.wConv=checkpoint['conv']
    model.wRect=checkpoint['rect']
    model.wHidden=checkpoint['wHidden']
    model.wHiddenBias=checkpoint['wHiddenBias']
    model.wNeu=checkpoint['wNeu']
    model.wNeuBias=checkpoint['wNeuBias']

    with torch.no_grad():
        model.mode = 'test'
        auc = []

        for i, (data, target) in enumerate(valid_loader):
            data = data.to(device)
            target = target.to(device)
            if reverse_mode:
                target_2 = torch.randn(int(target.shape[0]/2), 1)
                for i in range(target_2.shape[0]):
                    target_2[i] = target[2*i]
                target = target_2.to(device)
            
            # Forward Pass
            output = model(data)
            pred_sig = torch.sigmoid(output)
            pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
            labels = target.cpu().numpy().reshape(output.shape[0])

            auc.append(metrics.roc_auc_score(labels, pred))

        AUC_training = np.mean(auc)
        print(AUC_training)

    # Testing

    chipseq_test=Chip_test(dataset_names[1][i])
    test_data=chipseq_test.openFile()
    test_dataset=chipseq_dataset(test_data)
    batchSize=test_dataset.__len__()
    test_loader = DataLoader(dataset=test_dataset,batch_size=batchSize,shuffle=False)

    with torch.no_grad():
        model.mode='test'
        auc=[]
        
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            if reverse_mode:
                target_2=torch.randn(int(target.shape[0]/2),1)
                for i in range(target_2.shape[0]):
                    target_2[i]=target[2*i]
                target=target_2.to(device)
            # Forward pass
            output = model(data)
            pred_sig=torch.sigmoid(output)
            pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
            labels=target.cpu().numpy().reshape(output.shape[0])
            
            auc.append(metrics.roc_auc_score(labels, pred))
                
        AUC_test=np.mean(auc)
        print('AUC on test data = ',AUC_test)

    with open("./results/AUC_training.txt", "a") as file:
        file.write('TF Number ')
        file.write("%d" %(i+1))
        file.write(" - AUC Training : ")
        file.write("%d" %AUC_training)
        file.write("\n")
        file.write("---"*20)
        file.write("\n")
    file.close()

    with open("./results/AUC_testing.txt", "a") as file:
        file.write('TF Number ')
        file.write("%d" %(i+1))
        file.write(" - AUC Test : ")
        file.write("%d" %AUC_test)
        file.write("\n")
        file.write("---"*20)
        file.write("\n")
    file.close()