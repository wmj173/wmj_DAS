from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms


class DecodeBox(nn.Module):
    """YOLO解码器，将网络输出转换为预测框坐标

    Args:
        anchors (list): 锚框列表，每个锚框为(width, height)格式
        num_classes (int): 类别数量
        img_size (tuple): 输入图像尺寸 (width, height)
    """
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes  # 每个预测框的属性数量 (x,y,w,h,conf + class_probs)
        self.img_size = img_size

    def forward(self, input):
        """前向传播，将网络输出解码为预测框

        Args:
            input (Tensor): 网络输出特征图，形状为 [batch, num_anchors*(5+num_classes), height, width]

        Returns:
            Tensor: 解码后的预测框，形状为 [batch, num_predictions, 4+1+num_classes]
                   包含坐标(x1,y1,x2,y2)、置信度和类别概率
        """
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        # 计算特征图到原图的缩放比例
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # 根据特征图尺寸缩放锚框
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                         for anchor_width, anchor_height in self.anchors]

        # 调整输入形状以便处理
        prediction = input.view(batch_size, self.num_anchors,
                              self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 解码预测值
        x = torch.sigmoid(prediction[..., 0])  # 中心点x坐标偏移
        y = torch.sigmoid(prediction[..., 1])  # 中心点y坐标偏移
        w = prediction[..., 2]  # 宽度偏移
        h = prediction[..., 3]  # 高度偏移
        conf = torch.sigmoid(prediction[..., 4])  # 置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # 类别概率

        # 生成网格坐标
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        # 准备锚框尺寸
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        # 计算最终预测框坐标
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x  # 中心点x坐标
        pred_boxes[..., 1] = y.data + grid_y  # 中心点y坐标
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w  # 宽度
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h  # 高度

        # 缩放回原图尺寸并组合输出
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                          conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output

def letterbox_image(image, size):
    """保持图像长宽比进行缩放，并用灰色填充不足部分

    Args:
        image (PIL.Image): 输入图像
        size (tuple): 目标尺寸 (width, height)

    Returns:
        PIL.Image: 处理后的图像
    """
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    """将边界框坐标从网络输入尺寸转换回原始图像尺寸

    Args:
        top (np.array): 边界框顶部坐标
        left (np.array): 边界框左侧坐标
        bottom (np.array): 边界框底部坐标
        right (np.array): 边界框右侧坐标
        input_shape (tuple): 网络输入尺寸 (height, width)
        image_shape (tuple): 原始图像尺寸 (height, width)

    Returns:
        np.array: 转换后的边界框坐标 [x1,y1,x2,y2]
    """
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """计算两组边界框之间的IOU

    Args:
        box1 (Tensor): 边界框组1
        box2 (Tensor): 边界框组2
        x1y1x2y2 (bool): 边界框格式是否为[x1,y1,x2,y2]，否则为[center_x,center_y,w,h]

    Returns:
        Tensor: IOU值
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # 计算交集区域
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """非极大值抑制处理预测结果

    Args:
        prediction (Tensor): 模型预测输出
        num_classes (int): 类别数量
        conf_thres (float): 置信度阈值
        nms_thres (float): NMS阈值

    Returns:
        list: 每张图像的检测结果列表
    """
    # 将预测框从中心格式转换为角点格式
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 获取每个预测框的最高类别置信度
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # 根据置信度阈值筛选预测框
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # 组合检测结果 [x1,y1,x2,y2, obj_conf, class_conf, class_pred]
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        # 按类别进行NMS
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


def non_max_suppression_v2(prediction, num_classes, GT_bbox, conf_thres=0.5, nms_thres=0.4):
    """带GT框IOU计算的非极大值抑制

    Args:
        prediction (Tensor): 模型预测输出
        num_classes (int): 类别数量
        GT_bbox (Tensor): 真实边界框
        conf_thres (float): 置信度阈值
        nms_thres (float): NMS阈值

    Returns:
        tuple: (检测结果列表, 最大IOU值)
    """
    # 将预测框从中心格式转换为角点格式
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 计算预测框与GT框的最大IOU
        ious = bbox_iou(torch.tensor(GT_bbox).cuda().unsqueeze(0), image_pred[0:])
        max_iou = torch.max(ious)

        # 获取每个预测框的最高类别置信度
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # 根据置信度阈值筛选预测框
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # 组合检测结果 [x1,y1,x2,y2, obj_conf, class_conf, class_pred]
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        # 按类别进行NMS
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output, max_iou
