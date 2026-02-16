# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import logging
import os

import numpy as np
import torch
from sam3.perflib.compile import compile_wrapper
from sam3.perflib.masks_ops import mask_iou


try:
    from torch_generic_nms import generic_nms as generic_nms_cuda

    GENERIC_NMS_AVAILABLE = True
except ImportError:
    logging.debug(
        "Falling back to triton or CPU mask NMS implementation -- please install `torch_generic_nms` via\n\t"
        'pip uninstall -y torch_generic_nms; TORCH_CUDA_ARCH_LIST="8.0 9.0" pip install git+https://github.com/ronghanghu/torch_generic_nms'
    )
    GENERIC_NMS_AVAILABLE = False
    from sam3.perflib.triton.nms import nms_triton

_COMPILE_MASK_NMS_IOU = os.getenv("SAM3_COMPILE_MASK_NMS_IOU", "0") == "1"
if _COMPILE_MASK_NMS_IOU:
    # Compile the dense mask-IoU kernel used by mask NMS.
    # Keep this optional since compile has warmup overhead.
    _mask_iou_for_nms = compile_wrapper(
        mask_iou,
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=True,
        name="compiled_mask_iou_for_nms",
        make_contiguous=False,
        clone_output=False,
    )
else:
    _mask_iou_for_nms = mask_iou


def nms_masks(
    pred_probs: torch.Tensor,
    pred_masks: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Args:
      - pred_probs: (num_det,) float Tensor, containing the score (probability) of each detection
      - pred_masks: (num_det, H_mask, W_mask) float Tensor, containing the binary segmentation mask of each detection
      - prob_threshold: float, score threshold to prefilter detections (NMS is performed on detections above threshold)
      - iou_threshold: float, mask IoU threshold for NMS

    Returns:
     - keep: (num_det,) bool Tensor, indicating whether each detection is kept after score thresholding + NMS
    """
    # prefilter the detections with prob_threshold ("valid" are those above prob_threshold)
    is_valid = pred_probs > prob_threshold  # (num_det,)
    probs = pred_probs[is_valid]  # (num_valid,)
    num_valid = probs.numel()
    if num_valid <= 1:
        # no overlap suppression is needed for 0/1 valid detections
        return is_valid
    masks_binary = pred_masks[is_valid] > 0  # (num_valid, H_mask, W_mask)

    ious = _mask_iou_for_nms(masks_binary, masks_binary)  # (num_valid, num_valid)
    kept_inds = generic_nms(ious, probs, iou_threshold)

    keep = torch.zeros_like(is_valid)
    valid_det_inds = torch.nonzero(is_valid, as_tuple=False).squeeze(1)
    keep[valid_det_inds[kept_inds]] = True
    return keep


def generic_nms(
    ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5
) -> torch.Tensor:
    """A generic version of `torchvision.ops.nms` that takes a pairwise IoU matrix."""

    assert ious.dim() == 2 and ious.size(0) == ious.size(1)
    assert scores.dim() == 1 and scores.size(0) == ious.size(0)

    if ious.is_cuda:
        if GENERIC_NMS_AVAILABLE:
            return generic_nms_cuda(ious, scores, iou_threshold, use_iou_matrix=True)
        else:
            return nms_triton(ious, scores, iou_threshold)

    return generic_nms_cpu(ious, scores, iou_threshold)


def generic_nms_cpu(
    ious: torch.Tensor, scores: torch.Tensor, iou_threshold=0.5
) -> torch.Tensor:
    """
    A generic version of `torchvision.ops.nms` that takes a pairwise IoU matrix. (CPU implementation
    based on https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/nms/nms_cpu.py)
    """
    ious_np = ious.float().detach().cpu().numpy()
    scores_np = scores.float().detach().cpu().numpy()
    order = scores_np.argsort()[::-1]
    kept_inds = []
    while order.size > 0:
        i = order.item(0)
        kept_inds.append(i)
        inds = np.where(ious_np[i, order[1:]] <= iou_threshold)[0]
        order = order[inds + 1]

    return torch.tensor(kept_inds, dtype=torch.int64, device=scores.device)
