import torch

class WeightedCrossEntropy(torch.nn.Module):

    def __init__(self, device):
        super(WeightedCrossEntropy, self).__init__()
        self.device = device

    def forward(self, output, targets):

        """
        A custom loss function that weights loss for pixels closer to the output boundaries more heavily.

        Parameters:
            output (torch tensor): model output tensor of size (B, C, H, W)
            targets (torch): a tensor of target labels of size (B, :-1, H, W) and weight map of size (B, -1, H, W)
            
        Returns:
            loss (torch tensor): the weighted cross entropy loss
        """

        self.targets = targets
        self.output = output
        self.labels, self.weights = self.targets[:, :-1, :, :], self.targets[:, -1, :, :]

        # Calculate the cross entropy loss
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')(self.output, self.labels)
        self.loss = torch.mean(self.loss * self.weights)

        return self.loss
