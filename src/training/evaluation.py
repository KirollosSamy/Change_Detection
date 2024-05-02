import numpy as np
import torch
from torchmetrics.classification import BinaryJaccardIndex
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

def jaggard_batch(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.squeeze()
    target = target.squeeze()
    
    jaccard_index = BinaryJaccardIndex()
    total_jaggard = 0.0
    
    batch_size = pred.shape[0]
    
    for i in range(batch_size):
        jaccard = jaccard_index(pred[i], target[i]).item()
        if not np.isnan(jaccard):
            total_jaggard += jaccard
        else:
            total_jaggard += 1
        
    return total_jaggard / batch_size
    
def evaluate(model, dataloader, device='cpu', verbose=False):
    model.eval()
    
    total_jaggard = 0.0
    total_loss = 0.0
    
    criterion = torch.nn.BCEWithLogitsLoss()
            
    with torch.no_grad():
        for batch in tqdm(dataloader):
            A, B, delta = batch

            A = A.to(device)
            B = B.to(device)
            delta = delta.to(device)

            output = model(A, B)
            loss = criterion(output, delta)
            total_loss += loss.item()

            output = output.to(device='cpu')
            delta = delta.to(device='cpu')
            A = A.to(device='cpu')
            B = B.to(device='cpu')
            
            sigmoid_out = torch.sigmoid(output)
            change_map = torch.where(sigmoid_out > 0.5, torch.tensor(1), torch.tensor(0))
            
            if verbose:
                visualize_batch(A, B, delta, change_map)
            
            total_jaggard += jaggard_batch(change_map, delta)

    avg_loss = total_loss / len(dataloader)
    avg_jaggard = total_jaggard / len(dataloader)
    print(f'Evaluation Loss: {avg_loss:.6f}, Jaggard Index: {avg_jaggard:.6f}')

def visualize_batch(A, B, delta, change_map):    
    batch_size = A.shape[0]

    for i in range(batch_size):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(A[i].permute(1, 2, 0).numpy())
        plt.title('Image A')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(B[i].permute(1, 2, 0).numpy())
        plt.title('Image B')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(change_map[i].squeeze().numpy(), cmap='gray')
        plt.title('Model Output')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(delta[i].squeeze().numpy(), cmap='gray')
        plt.title('Delta')
        plt.axis('off')

        plt.show()