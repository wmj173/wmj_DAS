import cv2
import torch
import torchvision.transforms
from torchvision.transforms import transforms
from src.YoloV3.misc_functions import *

import torchvision.transforms as transforms



class Attention(object):
    """
    适配YOLOv5-YOLOv3模型与原有注意力计算类的适配器，包含数据增强功能
    """
    def __init__(self, model, ori_shape, final_shape, yolo_decodes, num_classes, conf_thres, nms_thres):
        """
        初始化适配器
        
        参数:
            model: YOLOv5-YOLOv3模型
            ori_shape: 原始图像形状
            final_shape: 最终图像形状
            yolo_decodes: YOLO解码器列表
            num_classes: 类别数量
            conf_thres: 置信度阈值
            nms_thres: NMS阈值
        """
        self.model = model
        self.feature = list()
        self.gradient = list()
        self.handlers = []
        
        # 共享Attention类的属性
        self.ori_shape = ori_shape
        self.final_shape = final_shape
        self.yolo_decodes = yolo_decodes
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        
        # 将模型设置为评估模式
        self.model.eval()
        
    def _get_features_hook(self, module, input, output):
        """PyTorch前向钩子函数"""
        self.feature.append(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        """PyTorch反向钩子函数"""
        self.gradient.append(output_grad[0])
        
    def _register_hook(self):
        """注册前向和反向传播的钩子函数"""
        self.feature = list()
        self.gradient = list()
        self.handlers = []

        self.handlers.append(
                self.model.model.model[15].register_forward_hook(self._get_features_hook)
            )

        self.handlers.append(
                self.model.model.model[15].register_backward_hook(self._get_grads_hook)
            )
    def remove_handlers(self):
        """移除所有注册的钩子处理器"""
        for handle in self.handlers:
            handle.remove()
            torch.cuda.empty_cache()

    def generate_grad_cam_plus_plus(self):
        """
        实现Grad-CAM++算法（去掉循环版本）
        """
        # 获取梯度和特征图（只处理第一个特征图）
        grad_val = self.gradient[0].clone().detach()
        feature_val = self.feature[0]

        # 计算Grad-CAM++权重
        # 公式: alpha = 1/Z * [grad^2 / (2*grad^2 + sum(grad^3 * A))]
        grad_2 = grad_val.pow(2)
        grad_3 = grad_val.pow(3)

        # 计算特征图的全局和
        feature_map_sum = feature_val.sum(dim=(2, 3), keepdim=True)

        # 防止除零错误的小常数
        eps = 1e-13

        # 计算alpha权重
        alpha = grad_2 / (2 * grad_2 + grad_3 * feature_map_sum + eps)

        # 将NaN值设为0
        alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)

        # 计算权重
        weights = alpha * torch.relu(grad_val)
        weights = weights.sum(dim=(2, 3), keepdim=True)

        # 生成CAM
        cam = torch.sum(weights * feature_val, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU操作
        cam = cam.squeeze(0).squeeze(0)  # 移除多余维度

        return cam

    def generate_grad_cam(self):
        """
        实现标准Grad-CAM算法
        """
        # 获取梯度和特征图（只处理第一个特征图）
        grad_val = self.gradient[0].clone().detach()
        feature_val = self.feature[0]

        # 计算全局平均池化权重
        weights = torch.mean(grad_val, dim=(2, 3), keepdim=True)

        # 生成CAM
        cam = torch.sum(weights * feature_val, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU操作
        cam = cam.squeeze(0).squeeze(0)  # 移除多余维度

        return cam

    def __call__(self, inputs, log_dir, retain_graph=True, use_augmentation=True):
        """
        计算注意力图，可选择是否使用数据增强

        参数:
            inputs: 输入字典，包含图像数据
            index: 索引
            retain_graph: 是否保留计算图
            use_augmentation: 是否使用数据增强（True=使用增强，False=不使用增强）

        返回:
            注意力图列表
        """
        # 注册钩子
        self._register_hook()
        try:
            # 获取输入图像
            img = inputs / 255  # 归一化到[0,1]范围
            raw_img = img.data.cpu().numpy()[0].transpose((1, 2, 0))
            img_ori = self.process_image(img)

            if not use_augmentation:
                # 不采取任何图像变换，直接使用原始图像
                outputs = self.model(img_ori)

                # 检查outputs的结构
                if isinstance(outputs, list) and len(outputs) >= 2:
                    # 如果是列表，第一个元素是检测结果，第二个元素是特征图
                    output = outputs[0]  # 检测结果 [1,22743,85]
                    feature_maps = outputs[1]  # 特征图列表
                elif isinstance(outputs, list) and len(outputs) == 3:
                    # 如果是特征图列表，需要解码
                    output_list = []
                    for k in range(len(outputs)):
                        output_list.append(self.yolo_decodes[k](outputs[k]))
                    output = torch.cat(output_list, 1)  # 拼接所有检测结果
                else:
                    # 单个检测结果张量
                    output = outputs

                # 计算置信度分数
                objectness = output[..., 4]  # 置信度分数
                if output.size(-1) > 5:  # 确保有类别分数
                    car_class_idx = 2  # car类别在CARLA_classes.txt中的索引为2
                    class_scores = output[..., 5 + car_class_idx]  # car类别的分数
                    bb = objectness * class_scores  # 组合分数（使用乘法更合理）
                else:
                    bb = objectness  # 只有置信度分数

                # 使用所有检测框的分数作为目标
                scores = torch.sum(bb)  # 对所有检测框的分数求和作为目标

                # 清零梯度并执行反向传播
                self.model.zero_grad()
                if scores.requires_grad:
                    scores.backward(retain_graph=retain_graph)

                # 检查是否有有效的梯度
                if len(self.gradient) > 0 and torch.sum(torch.abs(self.gradient[0])) > 0:
                    cam = self.generate_grad_cam_plus_plus()
                else:
                    print("警告：未能获取有效的梯度信息")

                CAM = cam
                cam_np = CAM.data.cpu().numpy()
                cam_np = np.maximum(cam_np, 0)  # ReLU操作
                cam_np = cv2.resize(cam_np, img_ori.shape[2:])  # 调整尺寸
                cam_np = cam_np - np.min(cam_np)  # 归一化
                cam_np = cam_np / np.max(cam_np) if np.max(cam_np) > 0 else cam_np
                self.show_cam_on_image(raw_img, cam_np,log_dir)
                return CAM,  scores

        finally:
            # 确保钩子被移除
            self.remove_handlers()
    def process_image(self, img):
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

    def show_cam_on_image(self, img, mask, log_dir):
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
        Image.fromarray(np.uint8(255 * cam)).save(os.path.join(log_dir, 'cam.jpg'))
        Image.fromarray(np.uint8(255 * mask)).save(os.path.join(log_dir, 'cam_b.jpg'))

