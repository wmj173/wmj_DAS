import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import warnings
import nmr_test as nmr
warnings.filterwarnings("ignore")



class MyDataset(Dataset):
    """
    自定义数据集类，用于加载和处理CARLA仿真环境生成的数据
    该类负责加载图像数据、车辆位置信息，并使用神经渲染器生成带纹理的图像
    """
    def __init__(self, data_dir, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False):
        """
        初始化数据集
        Args:
            data_dir: 数据目录路径
            img_size: 图像尺寸
            texture_size: 纹理尺寸
            faces: 3D模型的面片数据
            vertices: 3D模型的顶点数据
            distence: 距离阈值，用于过滤数据
            mask_dir: 掩码目录路径
            ret_mask: 是否返回掩码
        """
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        # 根据距离阈值过滤数据文件
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']  # 车辆位置
                cam_trans = data['cam_trans']  # 相机位置

                # 计算相机与车辆的相对位置
                cam_trans[0][0] = cam_trans[0][0] + veh_trans[0][0]
                cam_trans[0][1] = cam_trans[0][1] + veh_trans[0][1]
                cam_trans[0][2] = cam_trans[0][2] + veh_trans[0][2]

                veh_trans[0][2] = veh_trans[0][2] + 0.2

                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                # print(dis)
                # 如果距离小于阈值，则添加该文件
                if dis <= distence:
                    self.files.append(file)
        print(len(self.files))
        self.img_size = img_size
        
        # 初始化纹理参数
        textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures = torch.from_numpy(textures).cuda(device=0)
        self.faces_var = faces[None, :, :].cuda(device=0)
        self.vertices_var = vertices[None, :, :].cuda(device=0)
        
        # 初始化神经渲染器
        self.mask_renderer = nmr.NeuralRenderer(img_size=self.img_size).cuda()
        self.mask_renderer.renderer.renderer.camera_mode = "look_at"
        self.mask_renderer.renderer.renderer.light_direction = [0, 0, 1]
        self.mask_renderer.renderer.renderer.camera_up = [0, 0, 1]
        self.mask_renderer.renderer.renderer.background_color = [1, 1, 1]
        
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        # print(self.files)

    def set_textures(self, textures):
        """
        设置纹理参数
        Args:
            textures: 纹理张量
        """
        self.textures = textures

    def __getitem__(self, index):
        """
        获取单个数据项
        Args:
            index: 数据索引
        Returns:
            index: 数据索引
            total_img: 合成图像
            imgs_pred: 渲染图像
            mask: 掩码
        """
        # index = 5

        # print(index)
        # 加载数据文件
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file)
        img = data['img']           # 原始图像
        veh_trans = data['veh_trans']  # 车辆位置
        cam_trans = data['cam_trans']  # 相机位置
        
        # 计算相机与车辆的相对位置
        cam_trans[0][0] = cam_trans[0][0] + veh_trans[0][0]
        cam_trans[0][1] = cam_trans[0][1] + veh_trans[0][1]
        cam_trans[0][2] = cam_trans[0][2] + veh_trans[0][2]

        veh_trans[0][2] = veh_trans[0][2] + 0.2

        # 获取相机参数
        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)

        # 设置渲染器参数
        self.mask_renderer.renderer.renderer.eye = eye
        self.mask_renderer.renderer.renderer.camera_direction = camera_direction
        self.mask_renderer.renderer.renderer.camera_up = camera_up

        # 使用神经渲染器生成图像
        imgs_pred = self.mask_renderer.forward(self.vertices_var, self.faces_var, self.textures)


        # 处理原始图像
        img = img[:, :, ::-1]  # BGR转RGB
        img = cv2.resize(img, (self.img_size, self.img_size))  # 调整尺寸
        img = np.transpose(img, (2, 0, 1))  # HWC转CHW
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0)

        imgs_pred = imgs_pred / torch.max(imgs_pred)  # 归一化

        # 加载掩码文件
        mask_file = os.path.join(self.mask_dir, self.files[index][:-4] + '.png')
        mask = cv2.imread(mask_file)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        # 生成二值掩码
        mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        mask = torch.from_numpy(mask.astype('float32')).cuda()


        # 合成最终图像：原始图像*(1-掩码) + 渲染图像*掩码
        total_img = img * (1 - mask) + 255 * imgs_pred * mask

        return index, total_img.squeeze(0), imgs_pred.squeeze(0), mask

    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.files)


# if __name__ == '__main__':
#     obj_file = 'audi_et_te.obj'
#     vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, load_texture=True)
#     dataset = MyDataset('../data/phy_attack/train/', 608, 4, faces, vertices)
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=3,
#         shuffle=True,
#         # num_workers=2,
#     )
#
#     for img, car_box in loader:
#         print(img.size(), car_box.size())
# ß