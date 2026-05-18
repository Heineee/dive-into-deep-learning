import torch

def evalute_acc(model, data_iter,device):
  model.eval()
  acc_sum,n=0.0,0
  with torch.no_grad():
    for X, y in data_iter:
      X, y = X.to(device), y.to(device)
      y_hat=model(X)
      acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
      n+=y.shape[0]
  return acc_sum/n

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().sum().item()