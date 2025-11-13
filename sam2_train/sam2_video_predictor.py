# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

from models import register
import torch

from tqdm import tqdm

from sam2_train.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2_train.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames

import logging
import yaml
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from models import register
from tqdm import tqdm

from sam2_train.build_sam import build_sam2_video_predictor

import warnings
from collections import OrderedDict


logger = logging.getLogger(__name__)
from models.iou_loss import IOU
from typing import Any, Optional, Tuple

import sys
sys.path.append('..')


from torchvggish import vggish
from configs.vggish_config import cfg as vggish_cfg
from .matcher import HungarianMatcher

from sklearn.decomposition import NMF

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1, 2))
    union = (pred + target).sum(dim=(1, 2)) - inter

    epsilon = 1e-6
    iou = 1 - (inter / (union + epsilon))

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W
    
@register('sam2')
class SAM2VideoPredictor(SAM2Base):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        
        embed_dim=256
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.prompt_embed_dim = embed_dim

        self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        empty_weight = torch.ones(71)
        empty_weight[0] = 0.1
        self.CEloss = torch.nn.CrossEntropyLoss(weight=empty_weight.cuda())
        self.CEloss1 = torch.nn.CrossEntropyLoss()

        self.audio_MLP = MLP(embed_dim, embed_dim, embed_dim,3)
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.mlp = nn.Linear(embed_dim*3, embed_dim)
        self.mlp_audio = nn.Linear(embed_dim*3, embed_dim//2)
        self.seperate_audio = nn.Linear(embed_dim*3, embed_dim*3)
        self.token_all = nn.Linear(embed_dim*3, embed_dim)
        self.audio_to_text1 = nn.Linear(embed_dim*3, 6) #49408 is the vocb_size
        self.audio_to_texte = nn.Linear(embed_dim*3, 1024*6)
        self.text_a = nn.Linear(1024, embed_dim)
        self.down = nn.Linear(1280, embed_dim)
        self.aggregate =MLP(embed_dim*2, embed_dim, embed_dim,3)

        self.query_fea_W = nn.Embedding(5, embed_dim*3)
        self.query_embed_W = nn.Embedding(5, embed_dim*3)
        self.text_prompt_all = nn.Embedding(1, embed_dim*4)

        self.query_fea_H = nn.Embedding(5, 228)
        self.query_embed_H = nn.Embedding(5, 228)

        self.transformer_cross_attention_layers_W = nn.ModuleList()
        self.transformer_cross_attention_layers_H = nn.ModuleList()
        self.transformer_cross_attention_audio = nn.ModuleList()
        self.transformer_CA = nn.ModuleList()
        self.num_layers = 4
        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers_W.append(
                 CrossAttentionLayer(
                    d_model=embed_dim*3,
                    nhead=8,
                    dropout=0.0,
                )
            )

            self.transformer_cross_attention_layers_H.append(
                 CrossAttentionLayer(
                    d_model=228,
                    nhead=4,
                    dropout=0.0,
                )
            )

            self.transformer_cross_attention_audio.append(
                 CrossAttentionLayer(
                    d_model=768,
                    nhead=4,
                    dropout=0.0,
                )
            )
        for _ in range(self.num_layers//2):
            self.transformer_CA.append(
                 CrossAttentionLayer(
                    d_model=1024,
                    nhead=4,
                    dropout=0.0,
                )
            )
        
        self.matcher = HungarianMatcher(
            cost_class=5,
            cost_mask=5,
            cost_dice=5,
        )

        # get text information
        import json
        label_to_idx_path = AVSSBench/label2idx.json"
        with open(label_to_idx_path, 'r') as fr:
            label_to_pallete_idx = json.load(fr)
        labels = list(label_to_pallete_idx.keys())

        new_label = []
        for i in range(len(labels)):
            text = 'A {}'.format(labels[i])
            new_label.append(text)
        self.new_label = new_label
        self.imagebind = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebind.eval()

    def init_state(
        self,
        images,
        height, 
        width,
        audio,
        prompt,
        sparse,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        # get num, 3, 1024, 1024 feature
        ### here we do not utilize, as we already get the images
        # images, video_height, video_width = load_video_frames(
        #     video_path=video_path,
        #     image_size=self.image_size,
        #     offload_video_to_cpu=offload_video_to_cpu,
        #     async_loading_frames=async_loading_frames,
        #     compute_device=compute_device,
        # )
        inference_state = {}
        inference_state["images"] = images
        inference_state["audio"] = audio
        inference_state["num_frames"] = images.shape[1]
        inference_state["prompt"] = prompt
        inference_state["sparse"] = sparse
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = 224
        inference_state["video_width"] = 224
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        batch_size = 1
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=batch_size)
        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2VideoPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2VideoPredictor): The loaded model.
        """
        from sam2_train.build_sam import build_sam2_video_predictor_hf

        sam_model = build_sam2_video_predictor_hf(model_id, **kwargs)
        return sam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        #return len(inference_state["obj_idx_to_id"])
        return inference_state['images'].shape[0]

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
        sparse=None,
        prompt=None,
    ):
        """Add new points to a frame."""
        obj_output_dict = {}
        obj_temp_output_dict = {}
        obj_idx_all = []
        points_input = []
        for i in range(len(obj_id)):
            obj_idx = self._obj_id_to_idx(inference_state, obj_id[i])
            point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
            points_input.append([prompt[:,frame_idx,i,:].to(inference_state["device"]), sparse[:,frame_idx,:].to(inference_state["device"])])
            is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
            # whether to track in reverse time order
            if is_init_cond_frame:
                reverse = False
            else:
                reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
            obj_output_dict[obj_idx] = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict[obj_idx] = inference_state["temp_output_dict_per_obj"][obj_idx]
                # Add a frame to conditioning output if it's an initial conditioning frame or
                # if the model sees all frames receiving clicks/mask as conditioning frames.
            is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

            # Get any previously predicted mask logits on this object and feed it along with
            # the new clicks into the SAM mask decoder.
            obj_idx_all.append(obj_idx)
        prev_sam_mask_logits = None
        point_inputs_per_frame[frame_idx] = points_input

        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)

        prev_out_all = []
        for m in range(len(obj_idx_all)):
            prev_out = obj_temp_output_dict[obj_idx_all[m]][storage_key].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict[obj_idx_all[m]]["cond_frame_outputs"].get(frame_idx)
                if prev_out is None:
                    prev_out = obj_output_dict[obj_idx_all[m]]["non_cond_frame_outputs"].get(frame_idx)

            if prev_out is not None and prev_out[obj_idx_all[m]]["pred_masks"] is not None:
                device = inference_state["device"]
                prev_sam_mask_logits = prev_out[obj_idx[m]]["pred_masks"].to(device, non_blocking=True)
                # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
                prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
            prev_out_all.append(prev_sam_mask_logits)

        batch_size = 1
        current_out, _, audio_score = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=batch_size,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=points_input,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_out_all,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_num = len(obj_temp_output_dict)
        for i in range(len(obj_temp_output_dict)):
            out = {}
            for key, value in current_out.items():
                if value != None:
                    if len(value.shape) == 4:
                        total, _, p, p = value.shape
                        out[key] = value.reshape(total // obj_num, obj_num,1,p,p)[:,i,:,:,:]
                    elif key == 'audio_score':
                        out[key] = value.reshape(total // obj_num, obj_num,1,1)[:,i,:,:]
                    else:
                        total, p = value.shape
                        out[key] = value.reshape(total // obj_num, obj_num,1,p)[:,i,:,:]
                else:
                    out[key] = value
            obj_temp_output_dict[i][storage_key][frame_idx] = out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks, audio_score

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        print(consolidated_out["pred_masks_video_res"].shape)
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = len(inference_state["obj_idx_to_id"])
        bs = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(bs, batch_size, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
            "obj_ptr": torch.full(
                size=(bs, batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
             "audio_score":torch.full(
                size=(bs, batch_size, 1),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["device"],
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state, frame_idx
                        )
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][:, obj_idx : (obj_idx + 1), :] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_out['audio_score'][:, obj_idx : (obj_idx + 1), :]  = out['audio_score']
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[:, obj_idx : (obj_idx + 1), :] = obj_mask
                
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[:, obj_idx: (obj_idx + 1), :] = resized_obj_mask
            consolidated_out["obj_ptr"][:, obj_idx: (obj_idx + 1), :] = out["obj_ptr"]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if 'pred_masks_video_res' in consolidated_out.keys():
            consolidated_out['pred_masks_video_res'] = consolidated_out['pred_masks_video_res'].flatten(0,1)
        if "pred_masks" in consolidated_out.keys():
            consolidated_out["pred_masks"] = consolidated_out["pred_masks"].flatten(0,1).unsqueeze(1)
        consolidated_out["obj_ptr"] = consolidated_out["obj_ptr"].flatten(0,1)
        consolidated_out["audio_score"] = consolidated_out["audio_score"].flatten(0,1)
        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
        
        return consolidated_out

    def _get_empty_mask_ptr(self, inference_state, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # A dummy (empty) mask with a single object
        inference_state['images']
        batch_size = inference_state['images'].shape[0]
        mask_inputs = torch.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=torch.float32,
            device=inference_state["device"],
        )

        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = len(inference_state["obj_idx_to_id"])

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points_or_box` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                # to get the fused memory output
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = len(inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in processing_order:
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.

            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                audio_score = current_out['audio_score']
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                audio_score = current_out['audio_score']
            else:
                points_input = []
                point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_ids[0]]
                for i in range(len(obj_ids)):
                    points_input.append([inference_state['prompt'][:,frame_idx,i,:].to(inference_state["device"]), inference_state['sparse'][:,frame_idx,:].to(inference_state["device"])])
                point_inputs_per_frame[frame_idx] = points_input
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks, audio_score = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=points_input,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out
            ## Create slices of per-object outputs for subsequent interaction with each
            # # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # # Resize the output mask to the original video resolution (we directly use
            # # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks, audio_score

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            if len(inference_state["images"].shape) == 5:
                image = inference_state["images"][:,frame_idx,:,:,:].to(device).float()
            else:
                image = inference_state["images"][frame_idx,:,:,:].to(device).float()
            audio = inference_state["audio"][:,frame_idx,:].to(device).float()
            backbone_out = self.forward_image(image, audio)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        # here batch_size denote the number of objects
        expanded_image = image.unsqueeze(1).expand(-1, batch_size, -1, -1, -1).flatten(0,1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.unsqueeze(1).expand(
                -1, batch_size, -1, -1, -1
            ).flatten(0,1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.unsqueeze(1).expand(-1, batch_size, -1, -1, -1).flatten(0,1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        audio_score = current_out["audio_score"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "audio_score": audio_score,
        }
        return compact_current_out, pred_masks_gpu, audio_score

    def _run_memory_encoder(
        self, inference_state, frame_idx, batch_size, high_res_masks, is_mask_from_pts
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    def crop_and_resize_masked_image(self, image, mask, target_size=224, padding_factor=1):
        bs, T, C, H, W = image.shape
        
        result = []
        for b in range(bs):
            for t in range(T):
                current_image = image[b, t]
                current_mask = mask[b, t]
                
                y_indices, x_indices = torch.where(current_mask.sum(dim=0) > 0)
                if len(x_indices) != 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    
                    width = x_max - x_min
                    height = y_max - y_min
                    padding_x = int(width * padding_factor)
                    padding_y = int(height * padding_factor)
                    
                    x_min = max(0, x_min - padding_x)
                    x_max = min(W - 1, x_max + padding_x)
                    y_min = max(0, y_min - padding_y)
                    y_max = min(H - 1, y_max + padding_y)
                    cropped_image = current_image[:, y_min:y_max+1, x_min:x_max+1]

                    import torchvision.transforms as trans
                    resize = trans.Resize((target_size, target_size), antialias=True)
                    resized_image = resize(cropped_image)
                else:
                    resized_image = current_mask
                
                result.append(resized_image)
    
        return torch.stack(result).reshape(bs, T, C, target_size, target_size)

    def apply_gaussian_blur(self, image, bs = 2, kernel_size=75):
        padding = kernel_size // 2
        from einops import rearrange
        blurred_image = F.avg_pool2d(image.flatten(0,1), kernel_size, stride=1, padding=padding)
        return rearrange(blurred_image, '(b n) c h w -> b n c h w', b=bs)

    def set_input(self, input, ini_input, audio_paths, gt_mask, height, width):
        self.input = input.to(self.device)
        self.ini_input = ini_input.to(self.device)
        B, T, C, H, W = self.input.size()
        self.input = self.input.reshape(B, T, C, H, W)

        #### imagebind audio encoder
        with torch.no_grad():
            # spec_out = self.audio_encoder(self.spec_ini).reshape(B, T, -1)    # Bx128
            # self.spec = spec_out.detach().to(self.device)
            self.spec = data.load_and_transform_audio_data(audio_paths, self.device).detach()
            
        B, T, C_m, H_m, W_m = gt_mask.size()
        gt_mask = gt_mask.reshape(B, T, C_m, H_m, W_m)
        gt_mask = gt_mask.to(self.device)
        new_gt = torch.zeros_like(gt_mask)
        for i in range(B):
            classes = torch.unique(gt_mask[i])
            valid_classes = []
            for c in classes[1:]: # expext for background
                class_mask = (gt_mask[i] == c)
                area = class_mask.sum().item()  # 计算类别的像素数量（面积）
                if area > 20:
                    new_gt[i] = torch.where(class_mask, gt_mask[i], new_gt[i])
        self.gt_mask = new_gt

        self.height = height 
        self.width = width
        self.bs = B
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.text_all = data.load_and_transform_text(self.new_label, 'cuda')

    def forward(self):

        bs, T, _, _, _ = self.input.shape

        # 5 sequence
        ini_input = self.ini_input.reshape(-1, 3, 224, 224)  
        ini_input = F.interpolate(ini_input, size=(112, 112), mode='bilinear', align_corners=False)
        inputs = {
            ModalityType.VISION: ini_input.reshape(bs*T, 3, self.width//2, self.height//2),
            ModalityType.AUDIO:  self.spec[:,0,:,:,:],}
        with torch.no_grad():
            embeddings = self.imagebind(inputs)
        audio_fea = embeddings['audio_output'].detach().requires_grad_()  #bs*T,228,768
        audio_token_ini = embeddings['audio_token'].detach().requires_grad_().unsqueeze(1).repeat(1,5,1) 
        image_fea = rearrange(self.down(embeddings['vision_token']), '(b t) n -> b t n', b=self.bs)

        query_fea_W = self.query_fea_W.weight.unsqueeze(0).repeat(bs*T, 1, 1)
        query_pe_W = self.query_embed_W.weight.unsqueeze(0).repeat(bs*T, 1, 1)

        # spec_new bs,t,c -> bs*t,1,c
        for i in range(self.num_layers):
            query_W = self.transformer_cross_attention_layers_W[i](
                query_fea_W.permute(1,0,2), audio_fea.permute(1,0,2), #b,1,128
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=None, query_pos=query_pe_W.permute(1,0,2)
            )
        query_W = query_W.permute(1,0,2) #bs,5,128

        audio_bias = self.seperate_audio(query_W)
        audio_token = audio_token_ini + audio_bias
        for i in range(self.num_layers):
            audio_token = self.transformer_cross_attention_audio[i](
                audio_token.flatten(0,1).unsqueeze(1).permute(1,0,2), audio_fea.permute(1,0,2).repeat(1,5,1), #b,1,128
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=None, query_pos=None
            )
        audio_token = audio_token.permute(1,0,2) # bs*5,1,768

        # from audio embedding to text embedding
        prob_te = self.audio_to_text1(audio_token).squeeze(1)
        prob_all = self.audio_to_text1(embeddings['audio_token'])
        te_em = rearrange(self.audio_to_texte(audio_token), 'b l (n c) -> (b l) n c', c=1024)
        all_em = rearrange(self.audio_to_texte(embeddings['audio_token']), 'b (n c) -> b n c', c=1024)
        prob = torch.cat((prob_te, prob_all), dim=0)
        em = torch.cat((te_em, all_em), dim=0)

        inputs = {ModalityType.TEXT: (prob, em),}
        # with torch.no_grad():
        embeddings_t = self.imagebind(inputs)
        # self.text = embeddings['token]'][ModalityType.TEXT][:71]
        text_fea = rearrange(embeddings_t['text_output'][:bs*T*5,:], '(b t n) c -> (b t) n c',n=5, t=T, b=bs).detach().requires_grad_()
        self.audio_text_all = text_fea
        text_prompt_all = rearrange(embeddings_t['text_output'][bs*T*5:,:], '(b t n) c -> (b t) n c',n=1, t=T, b=bs).detach().requires_grad_()
        text_prompt = rearrange(torch.cat((text_fea, text_prompt_all), dim=-2), '(b t) n c -> b t n c',b=bs, t=T) # 4 5 77 6 1024
        text_prompt = self.text_a(text_prompt)

        # from audio embedding to text embedding
        query_W = self.mlp(query_W)
        spec_new = rearrange(self.mlp_audio(audio_fea).mean(dim=1), '(b t) c -> b t c',b=bs) #bs,T,128
        prompt = rearrange(self.audio_MLP(query_W), '(b t) n c -> b t n c',b=bs, t=T) #b, N, 256
        prompt_a =  rearrange(self.token_all(embeddings['audio_token'].unsqueeze(1)), '(b t) n c -> b t n c',b=bs) #b, N, 256
        prompt = torch.cat((prompt, prompt_a), dim=-2)

        total_prompt = self.aggregate(torch.cat((prompt, text_prompt), dim=-1))

        # initial state
        inference_state = self.init_state(self.input, self.height, self.width, spec_new, total_prompt, image_fea)
        ann_frame_idx = 0 
        # 5 is the binary token
        ann_obj_id = [0,1,2,3,4,5]
        
        _, out_obj_ids, out_mask_logits, score = self.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            sparse=image_fea,
            prompt=total_prompt,
        )

        output_masks_all = []
        output_query_masks = []
        score_all = []
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits, score in self.propagate_in_video(inference_state):
            output_masks = []
            video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id] = out_mask_logits.reshape(self.bs, 6, 224, 224)[:,i,:,:].unsqueeze(1)
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                output_masks.append(out_mask)
            outputs_masks = torch.cat(output_masks, dim=1).unsqueeze(1)
            output_query_masks.append(outputs_masks)
            score = score.reshape(bs, 1, len(ann_obj_id), 1, 1)
            score_all.append(score)
            
        self.all_masks = torch.cat(output_query_masks, dim=1)[:,:,:-1,:,:]
        self.binary_masks = torch.cat(output_query_masks, dim=1)[:,:,-1,:,:]
        self.reset_state(inference_state)

        # the background is all black
        mask_sigmoid = self.all_masks.sigmoid().reshape(bs, T, 5, self.height, self.width)
        result_mask = torch.zeros(bs, T, 5, 3, self.height, self.width).cuda()
        for m in range(5):
            current_mask = mask_sigmoid[:, :, m, :, :].unsqueeze(2).repeat(1, 1, 3, 1, 1)
            background_mask = 1 - current_mask
            blurred_background = self.apply_gaussian_blur(self.ini_input * background_mask, self.bs)
            result_mask[:, :, m, :, :, :] = self.ini_input * current_mask + blurred_background * background_mask

        result_mask = result_mask.reshape(-1, 3, 224, 224)  
        resized_mask = F.interpolate(result_mask, size=(112, 112), mode='bilinear', align_corners=False)
        # resized_mask = resized_mask.reshape(3, 5, 5, 3, 112, 112)
        inputs_v = {ModalityType.VISION: resized_mask.reshape(bs*T*5, 3, self.width//2, self.height//2),
                    ModalityType.AUDIO: audio_token.flatten(0,1).unsqueeze(1),}
        inputs_t = {ModalityType.TEXT: self.text_all,}

        # with torch.no_grad():
        embeddings_v = self.imagebind(inputs_v)
        with torch.no_grad():
            embeddings_t = self.imagebind(inputs_t)
        self.visual = embeddings_v[ModalityType.VISION]
        self.text = embeddings_t[ModalityType.TEXT]
        self.audio = embeddings_v[ModalityType.AUDIO]
        self.text_fea = embeddings_t['text_output'].detach()

    def infer(self, input, ini_input, spec, id):
        bs = 1
        T=1
        input = input.to(self.device).unsqueeze(0)
        ini_input = ini_input.reshape(-1, 3, 224, 224)  
        ini_input = F.interpolate(ini_input, size=(112, 112), mode='bilinear', align_corners=False)

        with torch.no_grad():
            spec = data.load_and_transform_audio_data(spec, id, self.device).detach()
        inputs = {
            ModalityType.VISION: ini_input.reshape(bs*T, 3, 112, 112),
            ModalityType.AUDIO:  spec.flatten(0,1),}
        with torch.no_grad():
            embeddings = self.imagebind(inputs)
        audio_fea = embeddings['audio_output']  #bs*T,228,768
        audio_token_ini = embeddings['audio_token'].unsqueeze(1).repeat(1,5,1) 
        image_fea = rearrange(self.down(embeddings['vision_token']), '(b t) n -> b t n', b=bs)

        query_fea_W = self.query_fea_W.weight.unsqueeze(0).repeat(bs*T, 1, 1)
        query_pe_W = self.query_embed_W.weight.unsqueeze(0).repeat(bs*T, 1, 1)
        query_fea_H = self.query_fea_H.weight.unsqueeze(0).repeat(bs*T, 1, 1)
        query_pe_H = self.query_embed_H.weight.unsqueeze(0).repeat(bs*T, 1, 1)

        # spec_new bs,t,c -> bs*t,1,c
        for i in range(self.num_layers):
            query_W = self.transformer_cross_attention_layers_W[i](
                query_fea_W.permute(1,0,2), audio_fea.permute(1,0,2), #b,1,128
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=None, query_pos=query_pe_W.permute(1,0,2)
            )
            
        query_W = query_W.permute(1,0,2) #bs,5,128

        audio_bias = self.seperate_audio(query_W)
        audio_token = audio_token_ini + audio_bias
        for i in range(self.num_layers):
            audio_token = self.transformer_cross_attention_audio[i](
                audio_token.flatten(0,1).unsqueeze(1).permute(1,0,2), audio_fea.permute(1,0,2).repeat(1,5,1), #b,1,128
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=None, query_pos=None
            )
        audio_token = audio_token.permute(1,0,2)

        # from audio embedding to text embedding
        prob_te = self.audio_to_text1(audio_token).squeeze(1)
        prob_all = self.audio_to_text1(embeddings['audio_token'])
        te_em = rearrange(self.audio_to_texte(audio_token).squeeze(1), 'b (n c) -> b n c', c=1024)
        all_em = rearrange(self.audio_to_texte(embeddings['audio_token']), 'b (n c) -> b n c', c=1024)
        prob = torch.cat((prob_te, prob_all), dim=0)
        em = torch.cat((te_em, all_em), dim=0)

        inputs = {ModalityType.TEXT: (prob, em),}
        with torch.no_grad():
            embeddings_t = self.imagebind(inputs)
        # self.text = embeddings['token]'][ModalityType.TEXT][:71]
        text_fea = rearrange(embeddings_t['text_output'][:bs*T*5,:], '(b t n) c -> (b t) n c',n=5, t=T, b=bs)
        self.audio_text_all = text_fea
        text_prompt_all = rearrange(embeddings_t['text_output'][bs*T*5:,:], '(b t n) c -> (b t) n c',n=1, t=T, b=bs)
        text_prompt = rearrange(torch.cat((text_fea, text_prompt_all), dim=-2), '(b t) n c -> b t n c',b=bs, t=T) # 4 5 77 6 1024
        text_prompt = self.text_a(text_prompt)

        # from audio embedding to text embedding
        query_W = self.mlp(query_W)
        spec_new = rearrange(self.mlp_audio(audio_fea).mean(dim=1), '(b t) c -> b t c',b=bs) #bs,T,128
        prompt = rearrange(self.audio_MLP(query_W), '(b t) n c -> b t n c',b=bs, t=T) #b, N, 256
        prompt_a =  rearrange(self.token_all(embeddings['audio_token'].unsqueeze(1)), '(b t) n c -> b t n c',b=bs) #b, N, 256
        prompt = torch.cat((prompt, prompt_a), dim=-2)

        total_prompt = self.aggregate(torch.cat((prompt, text_prompt), dim=-1))

        inference_state = self.init_state(input, 224, 224, spec_new, total_prompt, image_fea)
        ann_frame_idx = 0 
        ann_obj_id = [0,1,2,3,4,5]
        
        _, out_obj_ids, out_mask_logits, score = self.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            sparse=image_fea,
            prompt=total_prompt,
        )

        output_masks_all = []
        video_segments = {}  # video_segments contains the per-frame segmentation results
        score_all = []
        output_query_masks = []
        for out_frame_idx, out_obj_ids, out_mask_logits, score in self.propagate_in_video(inference_state):
            output_masks = []
            video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                slices = slice(i*bs, (i+1)*bs)
                video_segments[out_frame_idx][out_obj_id] = out_mask_logits[slices]
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                output_masks.append(out_mask)
            outputs_masks = torch.cat(output_masks, dim=1).unsqueeze(1)
            output_query_masks.append(outputs_masks)
            score = score.reshape(bs, 1, len(ann_obj_id), 1, 1)
            score_all.append(score)
            
        pred_mask = torch.cat(output_query_masks, dim=1)[:,:,-1,:,:]
         # the background is all black
        return pred_mask

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]).cuda()
        src_idx = torch.cat([src for (src, _) in indices]).cuda()
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]).cuda()
        tgt_idx = torch.cat([tgt for (_, tgt) in indices]).cuda()
        return batch_idx, tgt_idx
    
    @torch.jit.unused
    def _onnx_nested_tensor_from_tensor_list(self, tensor_list) -> NestedTensor:
        max_size = []
        for i in range(tensor_list[0].dim()):
            max_size_i = torch.max(
                torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
            ).to(torch.int64)
            max_size.append(max_size_i)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # m[: img.shape[1], :img.shape[2]] = False
        # which is not yet supported in onnx
        padded_imgs = []
        padded_masks = []
        for img in tensor_list:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

            m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
            padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
            padded_masks.append(padded_mask.to(torch.bool))

        tensor = torch.stack(padded_imgs)
        mask = torch.stack(padded_masks)

        return NestedTensor(tensor, mask=mask)

    def _max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def nested_tensor_from_tensor_list(self, tensor_list):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            import torchvision
            if torchvision._is_tracing():
                # nested_tensor_from_tensor_list() does not export well to ONNX
                # call _onnx_nested_tensor_from_tensor_list() instead
                return self._onnx_nested_tensor_from_tensor_list(tensor_list)

            # TODO make it support different-sized images
            max_size = self._max_by_axis([list(img.shape) for img in tensor_list])
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = [len(tensor_list)] + max_size
            b, c, h, w = batch_shape
            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return NestedTensor(tensor, mask)

    def attention_combination(self, v2t, a2t):
        # Learn attention weights
        alpha = torch.sigmoid(self.attention(torch.cat([v2t, a2t], dim=-1)))
        return alpha * v2t + (1 - alpha) * a2t

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # for individual query & mask
        from einops import rearrange
        self.loss_G = 0
        a2t = rearrange(torch.softmax(self.audio @ self.text.T, dim=-1), '(b t) c -> b t c',t=5)
        v2t = rearrange(torch.softmax(self.visual @ self.text.T, dim=-1), '(b t) c -> b t c',t=5)
        indices, label1, mask1 = self.matcher(self.gt_mask.flatten(0,1), a2t, v2t, self.all_masks.flatten(0,1))
        src_idx1 = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(label1, indices)])
        target_classes = torch.full((a2t.shape[0], 5, ), 0, dtype=torch.long, device=a2t.device)
        target_classes[src_idx1] = target_classes_o
        a2t_n = (self.audio @ self.text.T)
        v2t_n = (self.visual @ self.text.T)
        pred_mask = self.all_masks.flatten(0,1)
        gt_mask = torch.cat(mask1, dim=0)

        if len(src_idx1[0]) != 0:
            self.a2t = self.CEloss(a2t_n, target_classes.flatten(0,1)) / len(src_idx1[0])
            self.v2t = self.CEloss(v2t_n, target_classes.flatten(0,1)) / len(src_idx1[0])
            self.ce_sep = self.criterionBCE(pred_mask[src_idx1], gt_mask)
            self.iou_sep = _iou_loss(pred_mask[src_idx1], gt_mask)

            self.loss_G += 0.5*self.a2t
            self.loss_G += 0.5*self.v2t
            self.loss_G += self.ce_sep
            self.loss_G += self.iou_sep

            # total text
            text_prompt = torch.full((a2t.shape[0],5, 1024), 0, dtype=torch.float, device=a2t.device)
            for i in range(a2t.shape[0]):
                for j in range(5):
                    text_embedding = self.text_fea[target_classes[i][j], :]
                    text_prompt[i,j,:] = text_embedding

            audio_text = self.audio_text_all
            orig_text = text_prompt
            loss_fn = nn.MSELoss()
            self.loss_G += 0.3*loss_fn(audio_text, orig_text)
        
        # for binary mask, select class not equal to background to fuse
        target_masks = (self.gt_mask> 0).float().flatten(0,1).squeeze(1)
        pred_mask= self.binary_masks.flatten(0,1)
        self.ce_all = self.criterionBCE(pred_mask, target_masks)
        self.iou_all = _iou_loss(pred_mask, target_masks)
        self.loss_G += self.ce_all
        self.loss_G += self.iou_all
        self.loss_G.backward()

    def optimize_parameters(self):
            self.forward()
            self.optimizer.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
