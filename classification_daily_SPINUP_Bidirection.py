#!/usr/bin/env python
# coding: utf-8

# # random split since 11/13/2015 unidirection

# In[1]:

import pandas as pd
import numpy as np
import torch
from torch import nn

from sklearn.metrics import confusion_matrix
import seaborn as sn

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pd.options.mode.chained_assignment = None 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"'bidirection'"
class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length, hidden_size_1, nb_layers_1, drop_out_1, drop_out_2):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.hidden_size_1 = hidden_size_1
        # self.hidden_size_2 = hidden_size_2

        self.nb_layers_1 =nb_layers_1
        # self.nb_layers_2 = nb_layers_2 # number of LSTM layers (stacked)
 
        self.lstm_1 =nn.LSTM(input_size = n_features,
                              hidden_size = self.hidden_size_1, 
                              num_layers =self.nb_layers_1, 
                             bidirectional =True,
                              batch_first = True)  
                            
        self.dropout_1=nn.Dropout(p=drop_out_1)
        self.dropout_2=nn.Dropout(p=drop_out_2)

        self.l_linear = torch.nn.Linear(self.hidden_size_1 * 2, 2) # 2 if cross entropy; 1 if binary cross entropy or others

    def initialize_weights(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self,x):   
        batch_size, seq_len, n_features = x.size()
        h_t1, _ = self.lstm_1(x)
        h_t2_bi = self.dropout_1(h_t1)
        h_t2= h_t2_bi.view(batch_size, seq_len, -1)
        output_lst= [self.dropout_2(self.l_linear(h_t2[:,i,:].contiguous().view(batch_size,-1))) for i in range(seq_len)]
        output=torch.stack(output_lst, dim=1)
        
        return output



# In[3]:


"'Early  stopper'"
class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




# In[4]:


def confus_matrix(output,target,title):
    import seaborn as sn
    y_pred=torch.argmax(output, axis=-1).cpu().data.numpy()#, dim=1
    y_insitu=target.cpu().data.numpy()
    lstm_confu=confusion_matrix(y_insitu,y_pred)
    return sn.heatmap(lstm_confu,annot=True, annot_kws={"size": 16},fmt=".3g"), plt.ylabel('y_true'),    plt.xlabel('y_pred'),plt.title(title)


# # Results evaluation

# In[5]:


def output_loss_cvte(output,tensor_y):
    output_tmp=torch.permute(output,(0, 2,1))
    loss= criterion(output_tmp.to(device), tensor_y.to(device))#.float()
    return(loss.item())


# In[6]:


def custom_accuracy(output, target):
    output_sf=torch.argmax(output, axis=-1)#.cpu().data.numpy()#, dim=1
    train_acc = torch.sum(output_sf == target.to(device))
    final_train_acc = train_acc/target.size(0)#*target.size(1))
    accu=final_train_acc.cpu().data.numpy()
    return( accu)


# # Model configure

# In[7]:


def mymodel_configure_classification(para_dict,device):# if bidiretional true 2, if false 1
    mv_net = MV_LSTM(para_dict['n_features'],para_dict['seq_length'], para_dict['hidden_size_1'],  para_dict['nb_layers_1'], para_dict['drop_out_1'],para_dict['drop_out_2'])
    criterion=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mv_net.parameters(), lr=para_dict['learning_rate'])
    mv_net.to(device)
    return(mv_net,criterion,optimizer)





def train_validation_lstm_classification(para_dict,device,tensor_x_tr,tensor_y_tr,tensor_x_cv,tensor_y_cv):
# print(tensor_y_cv.shape)

    train_episodes=para_dict['train_episodes']
    batch_size=para_dict['batch_size']


#     df = pd.DataFrame(index=range(train_episodes),columns=('epoch', 'loss', 'accuracy'))
#     df_cv= pd.DataFrame(index=range(train_episodes),columns=('epoch', 'loss', 'accuracy'))

    early_stopper = EarlyStopper(patience=60, min_delta=0)

    mv_net,criterion,optimizer= mymodel_configure_classification(para_dict,device)

    for t in range(train_episodes):

        running_loss = []
        accuracy_tr=[]
        for b in range(0,len(tensor_x_tr),batch_size):
            inpt= tensor_x_tr[b:b+batch_size,:,:].to(device)
            target_tr= tensor_y_tr[b:b+batch_size,:]#.float()
            output_tr= mv_net(inpt.to(device))
            msf=torch.nn.Softmax(dim=1)
            target_122=target_tr[:,17:-15]
            output_122=output_tr[:,17:-15,:]
            target_tr_rmv2=target_122[target_122!=2]
            output_tr_rmv2= msf(output_122[target_122!=2,:])#.flatten()

            loss_tr = criterion(output_tr_rmv2.to(device),  target_tr_rmv2.to(device))#.float()

            #loss of train each batch
            running_loss.append(loss_tr.item())

            #accuracy of train each batch
            ac_s_tr=custom_accuracy(output_tr_rmv2 , target_tr_rmv2)



            accuracy_tr.append(ac_s_tr)
            loss_tr.backward()
            optimizer.step()        
            optimizer.zero_grad() 


        # #  train each epoch
        # print('train step : ' , t , 'loss : ' , np.mean(running_loss))
        # print('train step : ' , t , 'accuracy: ' , np.mean(accuracy_tr))



    #     # validation
        mv_net.eval()
        output_cv= mv_net(tensor_x_cv.to(device)) 
        target_122_cv=tensor_y_cv[:,17:-15]
        output_122_cv=output_cv[:,17:-15,:]
        target_cv_rmv2=target_122_cv[target_122_cv!=2]
        output_cv_rmv2=msf(output_122_cv[target_122_cv!=2,:])#.flatten()

        loss_cv = criterion(output_cv_rmv2.to(device),  target_cv_rmv2.to(device))#.float()
        accuracy_cv=custom_accuracy( output_cv_rmv2 ,target_cv_rmv2)
        mv_net.train()
#         print('cv step : ' , t , 'loss : ' ,loss_cv.item())
#         print('cv step : ' , t , 'accuracy: ' , accuracy_cv)
        if early_stopper.early_stop(loss_cv.item()):             
            break
    return(accuracy_cv,np.mean(accuracy_tr),mv_net,criterion)


#