import torch

class ImageDiff:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        
    def predict(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(A - B)
        delta = torch.where(diff >= self.threshold, torch.tensor(1), torch.tensor(0))
        return delta
    
class CVA:
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        
    def predict(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        diff = torch.norm(B-A, dim=1)
        delta = torch.where(diff >= self.threshold, torch.tensor(1), torch.tensor(0))
        return delta