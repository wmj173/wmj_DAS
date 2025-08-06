import sys
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms

from src.models.common import DetectMultiBackend
from src.yolo_utils.general import non_max_suppression


class YOLOv5:
    """
    YOLOv5适配器类，用于加载官方YOLOv5中的YOLOv3模型并实现注意力图计算
    """
    def __init__(self, weights='yolov3.pt', device='', conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000):
        """
        初始化YOLOv5适配器
        
        参数:
            weights (str): 模型权重文件路径
            device (str): 使用的设备 ('cpu', '0', '0,1,2,3' 等)
            conf_thres (float): 置信度阈值
            iou_thres (float): NMS的IoU阈值
            classes (list): 只保留特定类别，None表示保留所有类别
            max_det (int): 每张图像最大检测数量
        """
        # 初始化配置
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.max_det = max_det
        
        # 选择设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载模型
        self.model = DetectMultiBackend(weights, device=self.device)
        self.model.eval()  # 设置为评估模式
        
        # 提取模型信息
        self.stride = self.model.stride
        self.names = self.model.names  # 类别名称
        self.pt = self.model.pt  # PyTorch模型标志
        
        # 注意力图相关变量
        self.gradients = None
        self.activations = None
        self.target_layer = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """
        注册用于计算Grad-CAM的前向和反向传播钩子
        """
        # 为了简化，我们选择模型的一个特定层作为目标层
        # 通常是最后一个卷积层或特征提取部分的末尾
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            # YOLOv3模型结构
            backbone = self.model.model.model
            
            # 查找适合的目标层 (通常是最后的特征提取层)
            # 对于YOLOv3，这通常是darknet backbone的最后一个层
            self.target_layer = None
            
            # 遍历模型寻找合适的目标层
            for name, module in backbone.named_modules():
                if isinstance(module, nn.Conv2d) and name.startswith('model.9'):
                    self.target_layer = module
                    break
            
            if self.target_layer is not None:
                # 注册钩子
                self.target_layer.register_forward_hook(self._activation_hook)
                self.target_layer.register_full_backward_hook(self._gradient_hook)
    
    def _activation_hook(self, module, input, output):
        """
        前向传播钩子，用于存储激活值
        """
        self.activations = output.detach()
    
    def _gradient_hook(self, module, grad_input, grad_output):
        """
        反向传播钩子，用于存储梯度
        """
        self.gradients = grad_output[0].detach()
    
    def preprocess_image(self, img):
        """
        预处理图像用于模型输入
        
        参数:
            img: 输入图像 (可以是PIL.Image, numpy数组或OpenCV图像)
            
        返回:
            预处理后的张量
        """
        # 确保图像是RGB格式
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:  # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError("输入应该是PIL.Image或numpy数组")
        
        # 调整图像大小到模型输入尺寸
        img_size = self.model.stride.max() * max(self.model.pt['stride'] if isinstance(self.model.pt, dict) else [32])  # 确定输入尺寸
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        # 应用变换
        tensor = transform(img).unsqueeze(0)  # 添加批次维度
        return tensor.to(self.device)
    
    def compute_attention_map(self, img):
        """
        计算注意力图 (Grad-CAM)
        
        参数:
            img: 输入图像
            
        返回:
            注意力热图
        """
        # 预处理图像
        tensor = self.preprocess_image(img)
        
        # 清除之前的激活和梯度
        self.activations = None
        self.gradients = None
        
        # 前向传播
        with torch.enable_grad():
            tensor.requires_grad = True
            outputs = self.model(tensor)
            
            # 如果模型返回了多个输出，我们需要确定要使用哪个
            if isinstance(outputs, (list, tuple)):
                output = outputs[0]  # 使用第一个输出
            else:
                output = outputs
            
            # 计算模型置信度得分的总和 (通常是目标检测中的objectness score)
            # 这将作为我们的目标变量，我们将计算它关于目标层特征图的梯度
            scores = torch.max(output[..., 4:], dim=1)[0]  # 获取最大类别置信度
            score = torch.sum(scores)
            
            # 反向传播
            score.backward()
        
        # 检查是否成功获取了激活和梯度
        if self.activations is None or self.gradients is None:
            raise ValueError("未能获取层激活或梯度。请检查目标层设置。")
        
        # 计算梯度的全局平均池化
        weights = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 创建类激活图
        cam = torch.zeros_like(self.activations[0, 0]).to(self.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i]
        
        # 应用ReLU激活函数，只保留正值
        cam = torch.relu(cam)
        
        # 归一化CAM
        cam = cam - torch.min(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        
        # 调整CAM大小以匹配原始图像
        cam = cam.detach().cpu().numpy()
        if isinstance(img, Image.Image):
            original_size = img.size[::-1]  # (width, height) -> (height, width)
        else:
            original_size = img.shape[:2]  # (height, width)
            
        cam = cv2.resize(cam, (original_size[1], original_size[0]))
        
        return cam
    
    def detect(self, img):
        """
        使用模型进行目标检测
        
        参数:
            img: 输入图像
            
        返回:
            检测结果
        """
        # 预处理图像
        tensor = self.preprocess_image(img)
        
        # 推理
        with torch.no_grad():
            pred = self.model(tensor)
            
            # NMS
            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres,
                self.classes, agnostic=False, max_det=self.max_det
            )
        
        return pred
    
    def visualize_attention(self, img, cam, alpha=0.5):
        """
        将注意力热图可视化并叠加在原始图像上
        
        参数:
            img: 原始图像
            cam: 注意力热图
            alpha: 热图透明度
            
        返回:
            叠加了热图的可视化图像
        """
        # 确保图像是numpy数组
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img.copy()
        
        # 将热图转换为颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # 将热图从BGR转换为RGB (如果处理的是RGB图像)
        if img_array.ndim == 3 and img_array.shape[2] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 叠加热图和原始图像
        superimposed = heatmap * alpha + img_array * (1 - alpha)
        superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
        
        return superimposed 