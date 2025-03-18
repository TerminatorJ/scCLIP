import torch.nn as nn
import torch

class PairLoss(nn.Module):
    def __init__(self):
        '''
        calculating the binary cross entropy loss for the cell pairs
        '''
        super(PairLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # import pdb; pdb.set_trace()
        target = target.long()
        loss = self.loss(input, target)
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self, mask_way = "MT", n_token = None, cls_token = None, sep_token = None):
        '''
        mask_way: MT means mask the token. ME means mask the expression level
        
        '''
        super(MaskedMSELoss, self).__init__()
        self.mask_way = mask_way
        self.n_token = n_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        #get rid of the special token with <SEP> and <CLS>
        target[(target == self.cls_token) | (target == self.sep_token)] = -100
        try:
            # import pdb; pdb.set_trace()
            loss = self.loss(input, target)
            if torch.isnan(loss) or torch.isinf(loss):
                print("input:", input)
                print("target:", target)
                raise ValueError("Loss is NaN or Inf")
                
        except:
            print("Loss computation failed, assigning default loss value.")
            loss = torch.tensor(5, device=input.device)
        return loss
    
class MaskedMSE2DLoss(nn.Module):
    def __init__(self, sample_balance = False):
        super(MaskedMSE2DLoss, self).__init__()
        self.sample_balance = sample_balance

    def even_sample(self, pred, target):
        # Flatten matrices for easier indexing
        predictions_flat = pred.view(-1)
        true_labels_flat = target.reshape(-1)
        
        # Identify positive and negative indices
        positive_indices = (true_labels_flat == 1).nonzero(as_tuple=True)[0]
        negative_indices = (true_labels_flat == 0).nonzero(as_tuple=True)[0]
        
        # Sample indices
        num_samples = min(len(positive_indices), len(negative_indices))
        selected_pos_indices = torch.randperm(len(positive_indices))[:num_samples]
        selected_neg_indices = torch.randperm(len(negative_indices))[:num_samples]

        sampled_indices = torch.cat((positive_indices[selected_pos_indices], negative_indices[selected_neg_indices]))
        
        # Calculate loss using sampled indices
        sampled_predictions = predictions_flat[sampled_indices]
        sampled_true_labels = true_labels_flat[sampled_indices]
        return sampled_predictions, sampled_true_labels


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Here, the mask represents the masked tokens (with value != -100)
        # import pdb; pdb.set_trace()

        # Ensure valid_target is in the correct shape for gather
        if self.sample_balance:
            valid_input, valid_target = self.even_sample(input, target)
            mask_num = len(valid_target)
            try:
                spa_loss = F.binary_cross_entropy_with_logits(valid_input, valid_target.float(), reduction='mean')
                # spa_loss = loss / mask_num
                if torch.isnan(spa_loss) or torch.isinf(spa_loss):
                    print("valid_input:", valid_input)
                    print("valid_target:", valid_target)
                    raise ValueError("Loss is NaN or Inf")
            except:
                print("Loss computation failed, assigning default loss value.")
                spa_loss = torch.tensor(0.5, device=valid_input.device)

        return spa_loss