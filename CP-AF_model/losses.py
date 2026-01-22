import torch
import torch.nn as nn

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning loss, based on the official paper's implementation:
    https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07, base_temperature=0.07, device='cuda'):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels):
        """
        Calculates the supervised contrastive loss.

        Args:
            features (torch.Tensor): Projections from the model's projection head.
                Shape: [2 * batch_size, projection_dim]. It contains projections
                for two augmented views of each sample.
            labels (torch.Tensor): Ground truth labels for each sample.
                Shape: [2 * batch_size].

        Returns:
            torch.Tensor: A scalar loss value.
        """
        # Ensure the device is correct
        if features.device.type != self.device:
            features = features.to(self.device)
        if labels.device.type != self.device:
            labels = labels.to(self.device)

        batch_size = features.shape[0] // 2
        
        # Reshape labels to [2*B, 1] for broadcasting
        labels = labels.contiguous().view(-1, 1)

        # Create a mask to identify positive pairs
        # mask[i, j] is True if sample i and sample j have the same label.
        mask = torch.eq(labels, labels.T).float().to(self.device)
        
        # Compute cosine similarity between all pairs of projections.
        # anchor_dot_contrast[i, j] = sim(features[i], features[j])
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability, subtract the maximum logit
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create a mask to remove self-comparisons from the denominator
        # This is a matrix where diagonal elements are 0, others are 1.
        logits_mask = torch.ones_like(mask) - torch.eye(2 * batch_size, device=self.device)
        mask = mask * logits_mask

        # Compute log_prob
        # For each sample, the denominator is the sum of similarities with all other samples.
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute the mean log-likelihood over all positive pairs for each anchor
        # mask.sum(1) gives the number of positive pairs for each sample.
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Final loss is the negative mean of the mean log-likelihoods
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss