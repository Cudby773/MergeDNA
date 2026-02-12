import torch
import torch.nn as nn
from .merge_dna import MergeDNAEncoderModel

class LatentClassifier(nn.Module):
    """
    Wraps an encoder and attaches a small MLP classifier.
    Lazy-inits the classifier head based on the first batch's representation shape.
    """
    def __init__(self, encoder_model: MergeDNAEncoderModel, num_classes=1, use_cls_token=True, mlp_hidden=256, dropout=0.1):
        super().__init__()
        self.encoder_model = encoder_model
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.mlp_hidden = mlp_hidden
        self.dropout_p = dropout
        self.classifier = None
        self._head_inited = False
        
    def _build_head(self, hidden_dim: int):
        if self.num_classes == 1:
            head = nn.Sequential(
                nn.Linear(hidden_dim, self.mlp_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.mlp_hidden, 1)
            )
        else:
            head = nn.Sequential(
                nn.Linear(hidden_dim, self.mlp_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(self.mlp_hidden, self.num_classes)
            )        
        self.classifier = head
        self._head_inited = True
        return self.classifier


    def _init_head_from_rep(self, rep_tensor: torch.Tensor):
        # rep_tensor shape: [B, L, D] or [B, D]
        if rep_tensor.dim() == 3:
            hidden = rep_tensor.size(2)
        elif rep_tensor.dim() == 2:
            hidden = rep_tensor.size(1)
        else:
            raise RuntimeError("Unexpected encoder rep dim: " + str(rep_tensor.shape))
        head = self._build_head(hidden)
        head.to(rep_tensor.device)
        return head


    def forward(self, x):
        rep = self.encoder_model(x)  # expected [B, L, D] or [B, D]
        if not self._head_inited:
            self._init_head_from_rep(rep)

        if rep.dim() == 3:
            if self.use_cls_token:
                pooled = rep[:, 0, :]
            else:
                pooled = rep.mean(dim=1)
        else:
            pooled = rep

        logits = self.classifier(pooled)
        # normalize shape: binary single-logit -> [B], multiclass -> [B, C]
        if self.num_classes == 1:
            return logits.view(-1)
        return logits