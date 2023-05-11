import torch
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from model import DeepGAT,GAT
import hydra
from hydra import utils
from tqdm import tqdm
import mlflow
from utils import EarlyStopping,set_seed,log_artifacts


def train(loader,model,optimizer,device):
    model.train()
    loss_op = torch.nn.BCEWithLogitsLoss()
    total_loss = 0
    if model.cfg['layer_loss'] == 'supervised':
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out,hs,_ = model(data.x, data.edge_index)
            loss = loss_op(out, data.y)
            loss +=get_y_preds_loss(hs,data,loss_op)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
    else:
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out,_,_ = model(data.x, data.edge_index)
            loss = loss_op(out, data.y)
            total_loss += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader,model,device):
    model.eval()
    ys, preds,attentions = [], [],[]
    for data in loader:
        ys.append(data.y)
        out,_,attention = model(data.x.to(device), data.edge_index.to(device))
        attention = model.get_v_attention(data.edge_index,data.x.size(0),attention)
        attentions.append(attention)
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0, attentions

def get_y_preds_loss(hs,data,loss_op):
    y_pred_loss = torch.tensor(0, dtype=torch.float32,device=hs[0].device)
    for h in hs:
        h = h.mean(dim=1)
        y_pred_loss += loss_op(h,data.y)
    return y_pred_loss

def run(loader,model,optimizer,device,cfg):

    train_loader,test_loader = loader
    early_stopping = EarlyStopping(cfg['patience'],path=cfg['path'])

    for epoch in range(cfg['epochs']):
        loss_val = train(train_loader,model,optimizer,device)
        if early_stopping(loss_val,model,epoch) is True:
            break
    
    model.load_state_dict(torch.load(cfg['path']))
    test_acc,attentions = test(test_loader,model,device)
    return test_acc,early_stopping.epoch,attentions

@hydra.main(config_path='conf', config_name='config')
def main(cfg):
    torch.cuda.empty_cache()
    print(utils.get_original_cwd())
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment("output")
    mlflow.start_run()

    cfg = cfg[cfg.key]

    for key,value in cfg.items():
        mlflow.log_param(key,value)
    
    root = utils.get_original_cwd() + '/data/' + cfg['dataset']
    train_dataset = PPI(root, split='train')
    val_dataset = PPI(root, split='val')
    test_dataset = PPI(root, split='test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    loader =[train_loader,test_loader]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    artifacts,test_accs,epochs,attentions_l = {},[],[],[] 
    for i in tqdm(range(cfg['run'])):
        set_seed(i)
        if cfg['mode'] == 'original':
            model = GAT(cfg).to(device)
        else:
            model = DeepGAT(cfg).to(device)
             
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learing_late'])
        test_acc,epoch,attentions = run(loader,model,optimizer,device,cfg)

        test_accs.append(test_acc)
        epochs.append(epoch)
        attentions_l.append(attentions)
    
    acc_max_index = test_accs.index(max(test_accs))
    for j in range(len(attentions_l[acc_max_index])):
            artifacts[f"{cfg['dataset']}_{cfg['att_type']}_attention_{j}_L{cfg['num_layer']}.npy"] = attentions_l[acc_max_index][j]

    test_acc_ave = sum(test_accs)/len(test_accs)
    epoch_ave = sum(epochs)/len(epochs)
    log_artifacts(artifacts)

    mlflow.log_metric('epoch_mean',epoch_ave)
    mlflow.log_metric('test_acc_min',min(test_accs))
    mlflow.log_metric('test_acc_mean',test_acc_ave)
    mlflow.log_metric('test_acc_max',max(test_accs))
    
    mlflow.end_run()
    return test_acc_ave


    
if __name__ == "__main__":
    main()