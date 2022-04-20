import torch
from torch.utils.data import Dataset
from torch import nn
import transformers
import time
import copy
import matplotlib.pyplot as plt
import pickle

class KGEBERT(nn.Module):
    """Basic feedforward neural network"""
    def __init__(self, MODEL_PATH, kg_hidden=1500):
        """
        Input:
            - hidden_dim: hidden layer dimension. Assume two hidden layers have
                the same dimension

        A few key steps of the network:
            concatenation -> linear -> relu -> linear -> relu -> linear
        """
        super().__init__()
        self.bert = transformers.AutoModel.from_pretrained(MODEL_PATH)
        for param in self.bert.parameters():
            param.requires_grad = True
        if 'hidden_size' in self.bert.config.to_dict():
            embedding_dim = self.bert.config.to_dict()['hidden_size']
        else:
            embedding_dim = self.bert.config.to_dict()['dim']
        self.out = nn.Linear(embedding_dim*2, 2)

        self.fc1 = nn.Linear(1500,kg_hidden)
        nn.init.normal_(self.fc1.weight,mean=0,std=0.01)
        torch.nn.init.constant_(self.fc1.bias, 0)
        self.fc2 = nn.Linear(kg_hidden,embedding_dim)
        nn.init.normal_(self.fc2.weight,mean=0,std=0.01)
        torch.nn.init.constant_(self.fc2.bias, 0)
        self.ReLU = nn.ReLU()
        seq = [self.fc1, self.ReLU, self.fc2]
        self.seq = nn.Sequential(*seq)

#         self.seq = nn.Linear(1500,embedding_dim)
#         nn.init.normal_(self.seq.weight,mean=0,std=0.01)
#         torch.nn.init.constant_(self.seq.bias, 0)

    
    def forward(self, input_ids, token_type_ids, attention_mask, kg_embedding):
        res = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            )
        # (batch_size, num_token, 768/1024)
        sequence_out = res.last_hidden_state
        batch_size, num_token, _ = sequence_out.shape

        kg_out = self.seq(kg_embedding)

        # (batch_size, 768/1024)
        kg_out = kg_out.repeat(1,num_token).view(batch_size,num_token,-1)
        # (batch_size, num_token, 768/1024)

        cat_out = torch.cat((sequence_out, sequence_out*kg_out/((sequence_out.shape[-1])**0.5)), dim=2)
        # (batch_size, num_token, 2*768/1024)
        cat_out = self.out(cat_out)
        # (batch_size, num_token, 2)
        start_logit, end_logit = cat_out.split(1,dim=-1)
        # (batch_size, num_token, 1)
        start_logit = start_logit.squeeze()
        end_logit = end_logit.squeeze()
        # (batch_size, num_token)

        return start_logit, end_logit
        

def get_hyper_parameters():
    hidden_dim = [500,700]
    hidden_dim = [900]
    lr = [2e-5]
    weight_decay = [0.02] 
    return hidden_dim, lr, weight_decay




def calculate_loss(pred_start, pred_end, start, end):
    crossentropyloss=nn.CrossEntropyLoss()
    return crossentropyloss(pred_start,start) + crossentropyloss(pred_end,end)



def get_metrics(start_true_list,start_pred_list,end_true_list,end_pred_list):
    EM = 0
    AvNA = 0
    F1 = 0
    for start, pred_start, end, pred_end in zip(start_true_list,start_pred_list,end_true_list,end_pred_list):
        ans = set(range(pred_start,pred_end+1))
        f1 = 0
        for s,e in zip(start,end):
            if s == 0 and e == 0:
                tmp = 1 if pred_start == 0 and pred_end == 0 else 0
                f1 = max(f1,tmp)
                continue
            truth = set(range(s,e+1))
            intersection = len(truth.intersection(ans))
            if intersection == 0:
                f1 = max(f1,0)
            else:
                p = intersection/len(ans)
                r = intersection/len(truth)
                f1 = max(f1,2*p*r/(p+r))
        F1 += f1
        
        for s,e in zip(start,end):
            if pred_start == s and pred_end == e:
                EM += 1
                break

        if start[0] == 0 and end[0] == 0:
            if pred_start == 0 and pred_end == 0:
                AvNA += 1
        else:
            if pred_start > 0 or pred_end > 0:
                AvNA += 1


    
    EM /= len(start_true_list)
    AvNA /= len(start_true_list)
    F1 /= len(start_true_list)

    return F1, EM, AvNA





def train_model(net, trn_loader, val_loader, optim, scheduler, num_epoch=5,
        patience=10, collect_cycle=30, device='cpu', verbose=True):
        
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_metrics = None, [0,0,0]
    num_bad_epoch = 0

    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        # Training:
        net.train()
        for input_ids, token_type_ids, attention_mask, kg_embedding, start, end in trn_loader:
            num_itr += 1
            loss = None
            
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            kg_embedding = kg_embedding.to(device)
            start = torch.tensor(start).to(device)
            end = torch.tensor(end).to(device)
            
            pred_start, pred_end = net(input_ids, token_type_ids, attention_mask, kg_embedding)
            loss = calculate_loss(pred_start, pred_end, start, end)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ###################### End of your code ######################
            
            if num_itr % collect_cycle == 0:  # Data collection cycle
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
            ))

        # Validation:
        F1, EM, AvNA, loss = get_performance(net, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation F1: {:.4f}".format(F1))
            print("Validation EM: {:.4f}".format(EM))
            print("Validation AvNA: {:.4f}".format(AvNA))
            print("Validation loss: {:.4f}".format(loss))
        if F1 > best_metrics[0]:
            best_model = copy.deepcopy(net)
            best_metrics = [F1, EM, AvNA]
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1
        
        # early stopping
        if num_bad_epoch >= patience:
            break
        
        # learning rate scheduler
        scheduler.step()
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'metrics': best_metrics,
    }
    return best_model, stats




def get_performance(net, data_loader, device):
    """
    Evaluate model performance on validation set or test set.
    """
    net.eval()
    start_true_list = []
    start_pred_list = []
    end_true_list = []
    end_pred_list = []
    total_loss = [] # loss for each batch

    with torch.no_grad():
        for input_ids, token_type_ids, attention_mask, kg_embedding, start, end in data_loader:
            loss = None # loss for this batch
            
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            kg_embedding = kg_embedding.to(device)
            start_flat = torch.tensor([s[0] for s in start]).to(device)
            end_flat = torch.tensor([e[0] for e in end]).to(device)
            
            pred_start, pred_end = net(input_ids, token_type_ids, attention_mask, kg_embedding)
            loss = calculate_loss(pred_start, pred_end, start_flat, end_flat)

            pred_start = torch.argmax(pred_start, dim=1)
            pred_end_fix = torch.zeros_like(pred_start)

            _, res = torch.sort(pred_end, dim=1, descending=True,stable=True)
            for i, e in enumerate(res):
                for ind in e:
                    if ind >= pred_start[i]:
                        pred_end_fix[i] = ind
                        break           

            start_pred_list.append(pred_start.cpu())
            end_pred_list.append(pred_end_fix.cpu())

            start_true_list += start
            end_true_list += end

    total_loss.append(loss.item())
    start_pred_list = torch.cat(start_pred_list)
    end_pred_list = torch.cat(end_pred_list)
    # print(start_pred_list)
    # print(end_pred_list)
    F1, EM, AvNA = get_metrics(start_true_list,start_pred_list,end_true_list,end_pred_list)

    total_loss = sum(total_loss) / len(total_loss)
    
    return F1, EM, AvNA, total_loss



def plot_loss(stats,picname):
    """Plot training loss and validation loss."""
    with open(f"stats_{picname}.pickle","ab") as f:
        print("saving stats")
        pickle.dump(stats,f)
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.savefig(f"loss_{picname}.png")

        




