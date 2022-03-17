import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli

class ConvNet(nn.Module):
    def __init__(self, num_motif, motif_len, poolType, neuType, mode, droprate, lr, momentum_rate, 
                sigmaConv, sigmaNeu, beta1, beta2, beta3, device, reverse_complement_mode = False):
        super(ConvNet, self).__init__()
        self.poolType = poolType
        self.neuType = neuType
        self.mode = mode
        self.reverse_complement_mode = reverse_complement_mode
        self.droprate = droprate
        self.lr = lr
        self.momentum_rate = momentum_rate
        self.sigmaConv = sigmaConv
        self.sigmaNeu = sigmaNeu
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.device = device

        self.wConv = torch.randn(num_motif, 4, motif_len).to(device)
        torch.nn.init.normal_(self.wConv, mean = 0, std = self.sigmaConv)
        self.wConv.requires_grad = True

        self.wRect = torch.randn(num_motif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect = -self.wRect
        self.wRect.requires_grad = True

        if neuType == 'nohidden':
            if poolType == 'maxavg':
                self.wNeu = torch.randn(2*num_motif, 1).to(device)
            else:
                self.wNeu = torch.randn(num_motif, 1).to(device)
            self.wNeuBias = torch.randn(1).to(device)
            torch.nn.init.normal_(self.wNeu, mean=0, std = self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias, mean=0, std = self.sigmaNeu)
        
        else:
            if poolType == 'maxavg':
                self.wHidden = torch.randn(2*num_motif, 32).to(device)
            else:
                self.wHidden = torch.randn(num_motif, 32).to(device)
            self.wHiddenBias = torch.randn(32).to(device)
            self.wNeu = torch.randn(32, 1).to(device)
            self.wNeuBias = torch.randn(1).to(device)
            torch.nn.init.normal_(self.wNeu, mean=0, std = self.sigmaNeu)
            torch.nn.init.normal_(self.wNeuBias, mean=0, std = self.sigmaNeu)
            torch.nn.init.normal_(self.wHidden, mean=0, std = 0.3)
            torch.nn.init.normal_(self.wHiddenBias, mean=0, std = 0.3)

            self.wHidden.requires_grad = True
            self.wHiddenBias.requires_grad = True
        
        self.wNeu.requires_grad = True
        self.wNeuBias.requires_grad = True
    
    def divide_two_tensors(self, x):
        l = torch.unbind(x)

        list1 = [l[2*i] for i in range(int(x.shape[0]/2))]
        list2 = [l[2*i + 1] for i in range(int(x.shape[0]/2))]
        x1 = torch.stack(list1, 0)
        x2 = torch.stack(list2, 0)
        return x1, x2
    
    def forward_pass(self, x, mask=None, use_mask=False):
        conv = F.conv1d(x, self.wConv, bias = self.wRect, stride = 1, padding=0)
        rect = conv.clamp(min=0) # -> rectification
        maxPool, _ = torch.max(rect, dim=2)
        if self.poolType == 'maxavg':
            avgPool = torch.mean(rect, dim=2)
            pool = torch.cat((maxPool, avgPool), 1) # -> pool 순서가 논문과 다르지만 상관 없을것 같다!
        else:
            pool = maxPool
        if(self.neuType == 'nohidden'):
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.droprate, size=len(pool[0]))
                    mask = torch.from_numpy(mask).float().to(self.device)
                pooldrop = pool*mask
                out = pooldrop @ self.wNeu
                out.add_(self.wNeuBias)
            else:
                out = self.droprate*(pool @ self.wNeu)
                out.add_(self.wNeuBias)
        else:
            hid = pool @ self.wHidden
            hid.add_(self.wHiddenBias)
            hid = hid.clamp(min = 0) # rectification
            if self.mode == 'training':
                if not use_mask:
                    mask = bernoulli.rvs(self.droprate, size = len(hid[0]))
                    mask = torch.from_numpy(mask).float().to(self.device)
                hid_drop = hid*mask
                out = self.droprate*(hid@self.wNeu)
                out.add_(self.wNeuBias)
            else:
                out = self.droprate*(hid@self.wNeu)
                out.add_(self.wNeuBias)
        
        return out, mask
    
    def forward(self, x):
        if not self.reverse_complement_mode:
            out, _ = self.forward_pass(x)
        else:
            x1, x2 = self.divide_two_tensors(x)
            out1, mask = self.forward_pass(x1)
            out2, _ = self.forward_pass(x2, mask, True)
            out = torch.max(out1, out2)
        
        return out

class ConvNet_test(nn.Module):
    def __init__(self,nummotif,motiflen,poolType,neuType,mode,dropprob, learning_rate,learning_Momentum,sigmaConv,sigmaNeu,beta1,beta2,beta3, device, reverse_complemet_mode):
        super(ConvNet_test, self).__init__()
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
        maxPool, _ = torch.max(rect, dim=2)
        if self.poolType=='maxavg':
            avgPool= torch.mean(rect, dim=2)                          
            pool=torch.cat((maxPool, avgPool), 1)
        else:
            pool=maxPool
        if(self.neuType=='nohidden'):
            if self.mode=='training': 
                if  not use_mask:
                    mask=bernoulli.rvs(self.dropprob, size=len(pool[0]))
                    mask=torch.from_numpy(mask).float().to(self.device)
                pooldrop=pool*mask
                out=pooldrop @ self.wNeu
                out.add_(self.wNeuBias)
            else:
                out=self.dropprob*(pool @ self.wNeu)
                out.add_(self.wNeuBias)       
        else:
            hid=pool @ self.wHidden
            hid.add_(self.wHiddenBias)
            hid=hid.clamp(min=0)
            if self.mode=='training': 
                if  not use_mask:
                    mask=bernoulli.rvs(self.dropprob, size=len(hid[0]))
                    mask=torch.from_numpy(mask).float().to(self.device)
                hiddrop=hid*mask
                out=self.dropprob*(hid @ self.wNeu)
                out.add_(self.wNeuBias)
            else:
                out=self.dropprob*(hid @ self.wNeu)
                out.add_(self.wNeuBias) 
        return out,mask
       
    def forward(self, x):
        
        if not  self.reverse_complemet_mode:
            out,_=self.forward_pass(x)
        else:
            
            x1,x2=self.divide_two_tensors(x)
            out1,mask=self.forward_pass(x1)
            out2,_=self.forward_pass(x2,mask,True)
            out=torch.max(out1, out2)
        
        return out