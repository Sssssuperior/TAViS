# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 12544):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, targets, pred_class_a, pred_class_v, pred_mask):
        """More memory-friendly matching"""
        #bs, num_queries = outputs["pred_logits"].shape[:2]
        # here we set the video num and bs a single 'bs'
        bs, num_queries = pred_class_a.shape[0], pred_class_a.shape[1]

        indices = []
        matched_tgt_ids = []
        tgt_ids_all = [] 
        masks_all = []
        
        # Iterate through batch size
        for b in range(bs):

            out_prob_a = pred_class_a[b].squeeze(1) # [num_queries, num_classes]
            out_prob_v = pred_class_v[b].squeeze(1) 
            tgt_ids = torch.unique(targets[b])[1:] #expext background
            # new_value = torch.tensor([71], device=tgt_ids.device) 
            # tgt_ids = torch.cat([tgt_ids, new_value])
            tgt_ids_all.append(tgt_ids)
            cost_a = -out_prob_a[:, tgt_ids]
            cost_v = -out_prob_v[:, tgt_ids]
            
            out_mask = pred_mask[b].squeeze(1) # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b].to(out_mask)
            new_tgt_mask = torch.zeros(len(tgt_ids), 224, 224, device=out_mask.device)
            for i in range(len(tgt_ids)):
                mask = (tgt_mask == tgt_ids[i]).int()
                new_tgt_mask[i,:,:] = mask

            out_mask = out_mask[:, None]
            tgt_mask = new_tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                # Compute the dice loss betwen masks
                # print(out_mask.shape)
                # print(tgt_mask.shape)
                if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    # print(out_mask.shape[0])
                    # print(tgt_mask.shape[0])
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            C = (
                self.cost_class * cost_a
                + self.cost_class * cost_v
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()
            i, j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)))
            matched_ids = tgt_ids[j]
            save_mask = new_tgt_mask[j]
            masks_all.append(save_mask)
            matched_tgt_ids.append(matched_ids)

        return indices, tgt_ids_all, masks_all

    @torch.no_grad()
    def forward(self, targets, pred_class_a, pred_class_v, pred_mask):
        return self.memory_efficient_forward(targets, pred_class_a, pred_class_v, pred_mask)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
