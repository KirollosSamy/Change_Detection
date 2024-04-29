import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex

def jaggard_loss(pred: torch.Tensor, target: torch.Tensor):
    jaccard_index = BinaryJaccardIndex()
    total_loss = 0.0
    
    N = pred.shape[0]
    
    for i in range(N):
        jaccard = jaccard_index(pred[i], target[i]).item()
        if not np.isnan(jaccard):
            total_loss += 1 - jaccard
        
    return total_loss / N
    
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for A, B, delta in dataloader:
            output = model(A, B)
            loss = jaggard_loss(delta, output)
            total_loss += loss

    avg_loss = total_loss / len(dataloader)
    print(f'Evaluation Loss: {avg_loss:.4f}')
