import torch


def logits_to_mask(logits, threshold=0.5):
    # Apply sigmoid to convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Apply threshold to convert probabilities to binary mask
    mask = probabilities > threshold

    return (
        mask.float()
    )  # Convert to float tensor for further processing or visualization


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
