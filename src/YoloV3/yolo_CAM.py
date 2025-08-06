import torch
import torch.nn as nn
import numpy as np
import os
import sys
import yaml  # 新增：YAML解析库
import cv2
from PIL import Image
from collections import OrderedDict

from src.YoloV3.attention import Attention
from src.YoloV3.utils.utils import DecodeBox
from src.YoloV3.yolov5 import YOLOv5

# 添加项目根目录到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)



class YOLO(object):
    """
    兼容原始项目的YOLOv3接口，但内部使用YOLOv5-YOLOv3实现
    并沿用原有项目中的attention.py的注意力图计算方式
    """
    _defaults = {
        "model_path": r'E:\wmj\transPhyAtt-main\TrainingGT\ground-truth\best.pt',
        "anchors_path": 'models/hub/yolov3.yaml',
        "classes_path": 'YoloV3/utils/CARLA_classes.txt',
        "model_image_size": (608, 608, 1),
        "confidence": 0.25,
        "iou": 0.5,
        "cuda": True,
        "letterbox_image": False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "未识别的参数名称 '" + n + "'"

    def __init__(self, **kwargs):
        # 加载默认设置
        self.__dict__.update(self._defaults)

        # 使用配置覆盖默认设置
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 读取类别名称
        self.class_names = self._get_class()

        # # 读取锚框配置
        # self.anchors = self._get_anchors()

        # 读取锚框配置（从YAML文件）
        self.anchors = self._get_anchors_from_yaml()

        # 保存配置对象
        # self.config = config

        # 生成YOLOv5适配器和原有注意力计算方法
        self.generate()

    def _get_class(self):
        """从文件加载类别名称"""
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # def _get_anchors(self):
    #     """从文件加载锚框配置"""
    #     anchors_path = os.path.expanduser(self.anchors_path)
    #     with open(anchors_path) as f:
    #         anchors = f.readline()
    #     anchors = [float(x) for x in anchors.split(',')]
    #     return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    def _get_anchors_from_yaml(self):
        """从YAML文件加载锚框配置"""
        anchors_path = os.path.expanduser(self.anchors_path)
        try:
            with open(anchors_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            # 从YAML中提取anchors
            anchors = yaml_data.get('anchors', [])
            if not anchors:
                raise ValueError("YAML文件中没有找到anchors配置")

            # 将嵌套列表展平为一维列表
            flat_anchors = []
            for anchor_group in anchors:
                flat_anchors.extend(anchor_group)

            # 转换为numpy数组并reshape
            anchors_array = np.array(flat_anchors, dtype=np.float32)
            return anchors_array.reshape([-1, 3, 2])[::-1, :, :]

        except FileNotFoundError:
            raise FileNotFoundError(f"YAML文件未找到: {anchors_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML文件解析错误: {e}")

    def generate(self):
        """初始化模型"""
        self.num_classes = len(self.class_names)

        # 设置设备
        device = 'cuda' if torch.cuda.is_available() and self.cuda else 'cpu'

        # 使用配置中的权重文件
        # 从_defaults中获取权重文件路径，而不是从config对象
        weights_path = os.path.expanduser(self.model_path)  # 使用_defaults中的model_path
        if not os.path.exists(weights_path):
            raise ValueError(f"权重文件 {weights_path} 不存在")

        # 创建YOLOv5适配器
        self.adapter = YOLOv5(
            weights=weights_path,
            device=device,
            conf_thres=self.confidence,
            iou_thres=self.iou,
        )

        # 为YOLOv5模型创建YOLO解码器
        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], self.num_classes,
                          (self.model_image_size[1], self.model_image_size[0]))
            )

        # 创建多尺度注意力实例，使用原有的attention.py计算方法
        self.multi_attention = Attention(
            model=self.adapter.model,
            ori_shape=self.model_image_size,
            final_shape=self.model_image_size,
            yolo_decodes=self.yolo_decodes,
            num_classes=self.num_classes,
            conf_thres=self.confidence,
            nms_thres=self.iou
        )

        print(f'模型加载完成: {weights_path}')

