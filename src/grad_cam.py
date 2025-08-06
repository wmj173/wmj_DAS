import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import os

class GradCamPP():
    """
    Grad-CAM++算法实现类
    通过可视化神经网络的梯度来理解模型的决策过程
    """
    
    hook_a, hook_g = None, None  # 前向传播和反向传播的钩子变量
    
    hook_handles = []  # 钩子处理器列表
    
    def __init__(self, model, conv_layer, use_cuda=True):
        """
        初始化Grad-CAM++
        Args:
            model: 预训练模型
            conv_layer: 卷积层名称
            use_cuda: 是否使用CUDA
        """
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        # 注册前向传播钩子
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        # 注册反向传播钩子
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        """
        前向传播钩子函数
        Args:
            module: 模型模块
            input: 输入张量
            output: 输出张量
        """
        self.hook_a = output
        
    def clear_hooks(self):
        """
        清除所有钩子处理器
        """
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        """
        反向传播钩子函数
        Args:
            module: 模型模块
            grad_in: 输入梯度
            grad_out: 输出梯度
        """
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        """
        反向传播计算
        Args:
            scores: 模型输出分数
            class_idx: 类别索引
        """
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, class_idx, scores):
        """
        计算权重
        Args:
            class_idx: 类别索引
            scores: 模型输出分数
        Returns:
            权重值
        """
        self._backprop(scores, class_idx)
        
        grad_2 = self.hook_g.pow(2)
        grad_3 = self.hook_g.pow(3)
        alpha = grad_2 / (1e-13 + 2 * grad_2 + (grad_3 * self.hook_a).sum(axis=(2, 3), keepdims=True))

        # 在每个权重中应用像素系数
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(axis=(1, 2))
    
    def __call__(self, input, class_idx):
        """
        执行Grad-CAM++计算
        Args:
            input: 输入图像
            class_idx: 类别索引
        Returns:
            cam: CAM张量
            cam_np: CAM numpy数组
            pred: 预测概率
        """
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        pred = F.softmax(scores)[0, class_idx]
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        # rint(weights)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)  # ReLU操作
        cam_np = cv2.resize(cam_np, input.shape[2:])  # 调整尺寸
        cam_np = cam_np - np.min(cam_np)  # 归一化
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred

class GradCam():
    """
    Grad-CAM算法实现类
    通过可视化神经网络的梯度来理解模型的决策过程
    """
    
    hook_a, hook_g = None, None  # 前向传播和反向传播的钩子变量
    
    hook_handles = []  # 钩子处理器列表
    
    def __init__(self, model, conv_layer, use_cuda=True):
        """
        初始化Grad-CAM
        Args:
            model: 预训练模型
            conv_layer: 卷积层名称
            use_cuda: 是否使用CUDA
        """
        
        self.model = model.eval()
        self.use_cuda=use_cuda
        if self.use_cuda:
            self.model.cuda()
        
        # 注册前向传播钩子
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        
        self._relu = True
        self._score_uesd = True
        # 注册反向传播钩子
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))
        
    
    def _hook_a(self, module, input, output):
        """
        前向传播钩子函数
        Args:
            module: 模型模块
            input: 输入张量
            output: 输出张量
        """
        self.hook_a = output
        
    def clear_hooks(self):
        """
        清除所有钩子处理器
        """
        for handle in self.hook_handles:
            handle.remove()
    
    def _hook_g(self, module, grad_in, grad_out):
        """
        反向传播钩子函数
        Args:
            module: 模型模块
            grad_in: 输入梯度
            grad_out: 输出梯度
        """
        # print(grad_in[0].shape)
        # print(grad_out[0].shape)
        self.hook_g = grad_out[0]
    
    def _backprop(self, scores, class_idx):
        """
        反向传播计算
        Args:
            scores: 模型输出分数
            class_idx: 类别索引
        """
        loss = scores[:, class_idx].sum() # .requires_grad_(True)
        self.model.zero_grad()
        loss.backward(retain_graph=True)
    
    def _get_weights(self, class_idx, scores):
        """
        计算权重
        Args:
            class_idx: 类别索引
            scores: 模型输出分数
        Returns:
            权重值
        """
        self._backprop(scores, class_idx)
        
        return self.hook_g.squeeze(0).mean(axis=(1, 2))
    
    def __call__(self, input, class_idx):
        """
        执行Grad-CAM计算
        Args:
            input: 输入图像
            class_idx: 类别索引
        Returns:
            cam: CAM张量
            cam_np: CAM numpy数组
            pred: 预测概率
        """
        # print(input.shape)
        # if self.use_cuda:
        #     input = input.cuda()
        scores = self.model(input)
        # class_idx = torch.argmax(scores, axis=-1)
        pred = F.softmax(scores)[0, class_idx]
        # print(class_idx, pred)
        # print(scores)
        weights = self._get_weights(class_idx, scores)
        # print(input.grad)
        print(weights.unsqueeze(-1).unsqueeze(-1).shape)# (2048, 1, 1) self.hook_a.squeeze(0).shape  (2048,7,7)
        c = weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.squeeze(0)).sum(dim=0)
        
        # print(cam.shape)
        # self.clear_hooks()
        cam_np = cam.data.cpu().numpy()
        cam_np = np.maximum(cam_np, 0)  # ReLU操作
        cam_np = cv2.resize(cam_np, input.shape[2:])  # 调整尺寸
        cam_np = cam_np - np.min(cam_np)  # 归一化
        cam_np = cam_np / np.max(cam_np)
        return cam, cam_np, pred

class CAM:
    """
    CAM类，封装Grad-CAM功能
    """
    
    def __init__(self):
        """
        初始化CAM对象，加载预训练的ResNet50模型
        """
        model = models.resnet50(pretrained=True)
        self.grad_cam = GradCam(model=model, conv_layer='layer4', use_cuda=True)
        self.log_dir = "./"
        
    def __call__(self, img, index, log_dir, t_index=None):
        """
        执行CAM计算
        Args:
            img: 输入图像
            index: 索引
            log_dir: 日志目录
            t_index: 目标索引（可选）
        Returns:
            ret: CAM结果
            pred: 预测概率
        """
        self.log_dir = log_dir
        self.t_index = t_index
        img = img / 255  # 归一化到[0,1]范围
        raw_img = img.data.cpu().numpy()[0].transpose((1, 2, 0))  # 转换为HWC格式
        input = self.preprocess_image(img)  # 预处理图像
        target_index = [468,511,609,817,581,751,627]  # 目标类别索引列表
        if t_index==None:
            ret, mask, pred = self.grad_cam(input, target_index[index % len(target_index)])
        else:
            ret, mask, pred = self.grad_cam(input, t_index)
        # print(img.shape)
        self.show_cam_on_image(raw_img, mask)  # 显示CAM结果
        return ret, pred
        
    def preprocess_image(self, img):
        """
        预处理图像，进行标准化
        Args:
            img: 输入图像
        Returns:
            预处理后的图像
        """
        means = [0.485, 0.456, 0.406]  # ImageNet数据集的均值
        stds = [0.229, 0.224, 0.225]   # ImageNet数据集的标准差

        preprocessed_img = img
        for i in range(3):
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] - means[i]
            preprocessed_img[:, i, :, :] = preprocessed_img[:, i, :, :] / stds[i]
        input = preprocessed_img.requires_grad_(True)
        return input


    def show_cam_on_image(self, img, mask):
        """
        在图像上显示CAM结果
        Args:
            img: 原始图像
            mask: CAM掩码
        """
        # 应用颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        heatmap = np.float32(heatmap) / 255
        cam_pure = heatmap
        cam_pure = cam_pure / np.max(cam_pure)
        cam = np.float32(img) + heatmap  # 将热力图叠加到原始图像
        cam = cam / np.max(cam)
        if self.t_index==None:
            Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam.jpg'))
            Image.fromarray(np.uint8(255 * mask)).save(os.path.join(self.log_dir, 'cam_b.jpg'))
            
        else:
            Image.fromarray(np.uint8(255 * cam)).save(os.path.join(self.log_dir, 'cam_'+str(self.t_index)+'.jpg'))
        Image.fromarray(np.uint8(255 * cam_pure)).save(os.path.join(self.log_dir, 'cam_p.jpg'))
            
        # cv2.imwrite("cam.jpg", np.uint8(255 * cam))