import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F

from sklearn import metrics
from utils import logsampler, sqrtsampler, datasets, dataset_loader, test_dataset_loader
from network import ConvNet, ConvNet_test

print(torch.__version__)
print(torch.cuda.is_available())
    
def arg_parser():
    parser = argparse.ArgumentParser(description='Set hyperparameters or you can use default hyperparameter settings defined in the hyperparameter.json file')
    parser.add_argument('--TF', type=str, required=True, nargs='+', choices=['ARID3A', 'CTCFL', 'ELK1', 'FOXA1', 'GABPA', 'MYC', 'REST', 'SP1', 'USF1', 'ZBTB7A'], help='choose from [ARID3A / CTCFL / ELK1 / FOXA1 / GABPA / MYC / REST / SP1 / USF1 / ZBTB7A]')
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    TFs = args.TF
    print('TF Binding Prediction for ', end = '')
    for tf in TFs:
        print(tf, end = ' ')
    print()

    # Hyperparameters

    num_motif = 16 # number of motif detector (filter in CNN)
    motif_len = 24
    batch_size = 64
    reverse_mode=False
    num_grid_search = 5 # too small
    num_training_model = 5
    
    TFs2idx = {'ARID3A' : 0, 'CTCFL' : 1, 'ELK1' : 2, 'FOXA1' : 3, 'GABPA' : 4, 'MYC' : 5, 'REST' : 6, 'SP1' : 7, 'USF1' : 8, 'ZBTB7A' : 9}
    TFidx = [TFs2idx[TF] for TF in TFs]
    print('Corresponding Indices : ', end = '')
    print(TFidx)

    # Settings
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset path

    path = './data/encode/'
    all_dataset_names = datasets(path)

    for data_idx in TFidx:
        dataset_name = all_dataset_names[data_idx]

        train_dataset_path = dataset_name[0]
        test_dataset_path = dataset_name[1]
        print(train_dataset_path)
        print(test_dataset_path)

        name = train_dataset_path.split(path)[1].split("_AC")[0]
        print(name)

        train_dataloader, valid_dataloader, all_dataloader = dataset_loader(train_dataset_path, batch_size, reverse_mode)

        # Grid Search

        best_AUC=0
        learning_steps_list=[4000,8000,12000,16000,20000]

        for grid in range(num_grid_search):
            
            # randomly select hyperparameters
            pool_List=['max','maxavg']        
            random_pool=random.choice(pool_List)
            neuType_list=['hidden','nohidden']
            random_neuType=random.choice(neuType_list)
            dropoutList=[0.25,0.5,0.75] 
            dropprob=random.choice(dropoutList)
            learning_rate=logsampler(0.0005,0.05)
            momentum_rate=sqrtsampler(0.95,0.99)  
            sigmaConv=logsampler(10**-7,10**-3)   
            sigmaNeu=logsampler(10**-5,10**-2) 
            beta1=logsampler(10**-15,10**-3)
            beta2=logsampler(10**-10,10**-3)
            beta3=logsampler(10**-10,10**-3)

            model_auc=[[],[],[]]

            for idx in range(3):
                model = ConvNet(num_motif,motif_len,random_pool,random_neuType,'training',dropprob,learning_rate,momentum_rate,sigmaConv,sigmaNeu,beta1,beta2,beta3, device, reverse_complemet_mode=reverse_mode).to(device)
                if random_neuType=='nohidden':
                    optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias], lr=model.learning_rate,momentum=model.momentum_rate,nesterov=True)

                else:
                    optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias,model.wHidden,model.wHiddenBias], lr=model.learning_rate,momentum=model.momentum_rate,nesterov=True)

                train_loader=train_dataloader[idx]
                valid_loader=valid_dataloader[idx]

                learning_steps=0

                while learning_steps<=20000:
                    model.mode='training'
                    auc=[]
                    for i, (data, target) in enumerate(train_loader):
                        data = data.to(device)
                        target = target.to(device)
                        if model.reverse_complemet_mode:
                            target_2=torch.randn(int(target.shape[0]/2),1)
                            for i in range(target_2.shape[0]):
                                target_2[i]=target[2*i]
                            target=target_2.to(device)
                        
                        # Forward pass
                        output = model(data)
                        
                        if model.neuType=='nohidden':
                            loss = F.binary_cross_entropy(torch.sigmoid(output),target)+model.beta1*model.wConv.norm()+model.beta3*model.wNeu.norm()
                        else:
                            loss = F.binary_cross_entropy(torch.sigmoid(output),target)+model.beta1*model.wConv.norm()+model.beta2*model.wHidden.norm()+model.beta3*model.wNeu.norm()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        learning_steps+=1
            
                        if learning_steps% 4000==0:
                            with torch.no_grad():
                                model.mode='test'
                                auc=[]
                                for i, (data, target) in enumerate(valid_loader):
                                    data = data.to(device)
                                    target = target.to(device)
                                    if model.reverse_complemet_mode:
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
                                model_auc[idx].append(np.mean(auc))
                                print('Grid ', grid+1, ' with training fold ', idx+1, ' & learning steps ',learning_steps_list[len(model_auc[idx])-1], ' showed AUC of ' ,np.mean(auc))
            
            print('----------------------------------------------------------------------------------------------------')

            for n in range(5):
                AUC=(model_auc[0][n]+model_auc[1][n]+model_auc[2][n])/3
                if AUC>best_AUC:
                    best_AUC=AUC
                    best_learning_steps=learning_steps_list[n]
                    best_LearningRate=model.learning_rate
                    best_LearningMomentum=model.momentum_rate
                    best_neuType=model.neuType
                    best_poolType=model.poolType
                    best_sigmaConv=model.sigmaConv
                    best_dropprob=model.dropprob
                    best_sigmaNeu=model.sigmaNeu
                    best_beta1=model.beta1
                    best_beta2=model.beta2
                    best_beta3=model.beta3
                    at_grid = grid+1

        # Save The Best Hyperparameters

        print('best pooling layer type = ', best_poolType)
        print('best neural network type = ', best_neuType)
        print('best AUC = ', best_AUC)
        print('best learning_steps = ', best_learning_steps)
        print('best learning rate = ', best_LearningRate)
        print('best momentum = ', best_LearningMomentum)
        print('best sigmaConv = ', best_sigmaConv)
        print('best dropprob = ', best_dropprob)
        print('best sigmaNeu = ', best_sigmaNeu)
        print('best beta1 = ', best_beta1)
        print('best beta2 = ', best_beta2)
        print('best beta3 = ', best_beta3)
        print('At grid ', at_grid)

        hyperparameters = {'pool_type': best_poolType,
                        'neu_type':best_neuType,
                        'learning_steps':best_learning_steps,
                        'learning_rate':best_LearningRate, 
                        'momentum':best_LearningMomentum,
                        'sigmaConv':best_sigmaConv,
                        'dropprob':best_dropprob,
                        'sigmaNeu':best_sigmaNeu,
                        'beta1':best_beta1, 
                        'beta2':best_beta2,
                        'beta3':best_beta3}

        torch.save(hyperparameters, './Hyperparameters/' + name + '.pth') 

        # Model Training

        best_AUC=0

        best_hyperparameters = torch.load('./Hyperparameters/' + name + '.pth')
        best_poolType=best_hyperparameters['pool_type']
        best_neuType=best_hyperparameters['neu_type']
        best_learning_steps=best_hyperparameters['learning_steps']
        best_LearningRate=best_hyperparameters['learning_rate']
        best_dropprob=best_hyperparameters['dropprob']
        best_LearningMomentum=best_hyperparameters['momentum']
        best_sigmaConv=best_hyperparameters['sigmaConv']
        best_sigmaNeu=best_hyperparameters['sigmaNeu']
        best_beta1=best_hyperparameters['beta1']
        best_beta2=best_hyperparameters['beta2']
        best_beta3=best_hyperparameters['beta3']

        learning_steps_list=[4000,8000,12000,16000,20000]

        for model_number in range(num_training_model):

            model = ConvNet_test(num_motif,motif_len,best_poolType,best_neuType,'training',best_learning_steps,best_LearningRate,best_LearningMomentum,best_sigmaConv,best_dropprob,best_sigmaNeu,best_beta1,best_beta2,best_beta3,device,reverse_complemet_mode=False).to(device)

            if model.neuType=='nohidden':
                optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias], lr=model.learning_rate,momentum=model.momentum_rate,nesterov=True)
            else:
                optimizer = torch.optim.SGD([model.wConv,model.wRect,model.wNeu,model.wNeuBias,model.wHidden,model.wHiddenBias], lr=model.learning_rate,momentum=model.momentum_rate,nesterov=True)

            train_loader=all_dataloader
            valid_loader=all_dataloader
            learning_steps=0

            while learning_steps<=best_learning_steps:
                for i, (data, target) in enumerate(train_loader):
                    data = data.to(device)
                    target = target.to(device)
                    if model.reverse_complemet_mode:
                        target_2=torch.randn(int(target.shape[0]/2),1)
                        for i in range(target_2.shape[0]):
                            target_2[i]=target[2*i]
                        target=target_2.to(device)

                    # Forward pass
                    output = model(data)
                    
                    if model.neuType=='nohidden':
                        loss = F.binary_cross_entropy(torch.sigmoid(output),target)+model.beta1*model.wConv.norm()+model.beta3*model.wNeu.norm()
                    else:
                        loss = F.binary_cross_entropy(torch.sigmoid(output),target)+model.beta1*model.wConv.norm()+model.beta2*model.wHidden.norm()+model.beta3*model.wNeu.norm()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    learning_steps+=1
                    
            with torch.no_grad():
                model.mode='test'
                auc=[]
                for i, (data, target) in enumerate(valid_loader):
                    data = data.to(device)
                    target = target.to(device)
                    if model.reverse_complemet_mode:
                        target_2=torch.randn(int(target.shape[0]/2), 1)
                        for i in range(target_2.shape[0]):
                            target_2[i]=target[2*i]
                        target=target_2.to(device)

                    # Forward pass
                    output = model(data)
                    pred_sig=torch.sigmoid(output)
                    pred=pred_sig.cpu().detach().numpy().reshape(output.shape[0])
                    labels=target.cpu().numpy().reshape(output.shape[0])
                    auc.append(metrics.roc_auc_score(labels, pred))

                AUC_training=np.mean(auc)
                print('AUC for model ', model_number+1,' = ', AUC_training, ' while best = ', best_AUC)
                if AUC_training > best_AUC:
                    state = {'conv': model.wConv,
                            'rect':model.wRect,
                            'wHidden':model.wHidden,
                            'wHiddenBias':model.wHiddenBias,
                            'wNeu':model.wNeu,
                            'wNeuBias':model.wNeuBias}
                    torch.save(state, './Models/' + name + '.pth')
                    best_AUC = AUC_training

        # Training Performance

        checkpoint = torch.load('./Models/'+ name + '.pth')
        model = ConvNet_test(num_motif,motif_len,best_poolType,best_neuType,'test',best_learning_steps,best_LearningRate,best_LearningMomentum,best_sigmaConv,best_dropprob,best_sigmaNeu,best_beta1,best_beta2,best_beta3,device,reverse_complemet_mode=reverse_mode).to(device)
        model.wConv=checkpoint['conv']
        model.wRect=checkpoint['rect']
        model.wHidden=checkpoint['wHidden']
        model.wHiddenBias=checkpoint['wHiddenBias']
        model.wNeu=checkpoint['wNeu']
        model.wNeuBias=checkpoint['wNeuBias']

        with torch.no_grad():
            model.mode='test'
            auc=[]
            
            for i, (data, target) in enumerate(valid_loader):
                data = data.to(device)
                target = target.to(device)
                if model.reverse_complemet_mode:
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
                    
            AUC_training=np.mean(auc)
            print(AUC_training)

        # Testing

        test_loader = test_dataset_loader(test_dataset_path, motif_len)

        with torch.no_grad():
            model.mode='test'
            auc=[]
            
            for i, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                if model.reverse_complemet_mode:
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
            print('AUC on test data = ', AUC_test)

        # write results

        with open("./results/AUC_training.txt", "a") as file:
            file.write('TF : ')
            file.write(name)
            file.write(" - AUC Train : ")
            file.write(str(round(AUC_training, 3)))
            file.write("\n")
            file.write("---"*20)
            file.write("\n")
        file.close()

        with open("./results/AUC_testing.txt", "a") as file:
            file.write('TF : ')
            file.write(name)
            file.write(" - AUC Test : ")
            file.write(str(round(AUC_test, 3)))
            file.write("\n")
            file.write("---"*20)
            file.write("\n")
        file.close()

if __name__ == '__main__':
    main()

