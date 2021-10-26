#!/usr/bin/env python
# coding: utf-8

from collections import Sequence

import gc
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import torchvision
import cv2


class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.output = F.softmax(self.logits, dim=1)
        self.probs, self.ids= self.output.sort(dim=1, descending=True) 
        return  self.probs, self.ids

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def clear_mem(self):
        del self.logits, self.model
        del self.output,self.probs,self.ids
        self.remove_hook()
        # gc.collect()
        # torch.cuda.empty_cache()

class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        # print(len(pool))
        if target_layer in pool.keys():
            # print(repr(pool[target_layer]))
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)

        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

class CAM(_BaseWrapper):
    """
    "CAM
    """
    def __init__(self, model, candidate_layers=None):
        super(CAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            # print(target_layer)
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer, label_count):

        ## top k class probability
        topk_prob = self.probs.squeeze().tolist()[:label_count]
        topk_arg = self.ids.squeeze().tolist()[:label_count]
        # print(topk_prob)
        # print(topk_arg)

        # Get softmax weight
        params = list(self.model.parameters())
        weights = torch.from_numpy(np.squeeze(params[-2].data.cpu().numpy())).cuda()

        #self.logit, self.probs, self.ids
        # print("\nWEIGHTS SHAPE")
        # print(weights.shape)
        # print("\nFMAP SHAPE")

        #  Get feature map
        fmaps = self._find(self.fmap_pool, 'base_model.layer4')
        B, C, H, W = fmaps.shape

        # print(fmaps.shape)
        cam = torch.matmul(weights, fmaps.resize(B,C,H*W))

        # print(cam.shape)
        
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val

        topk_cam = cam.view(1, -1, H, W)[0,topk_arg]
        # print(topk_cam.shape)
        topk_cam = F.interpolate(topk_cam.unsqueeze(0), (self.image_shape[0],self.image_shape[1]), mode="bilinear", align_corners=False).squeeze(0)
        
        topk_cam = torch.split(topk_cam, 1)
        # print(topk_cam[0].shape)
        return topk_cam