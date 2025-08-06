from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.misc
import tqdm
import math
import os
import torch

import neural_renderer


#############
### Utils ###
#############

def convert_as(src, trg):
    """
    将源张量转换为目标张量的类型和设备
    Args:
        src: 源张量
        trg: 目标张量
    Returns:
        转换后的张量
    """
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src


def get_params(carlaTcam, carlaTveh):  
    """
    根据CARLA仿真环境中的相机和车辆位置计算渲染参数
    Args:
        carlaTcam: 相机位置和角度 (tuple of 2*3)
        carlaTveh: 车辆位置和角度 (tuple of 2*3)
    Returns:
        eye: 相机位置
        camera_direction: 相机方向
        camera_up: 相机上方向
    """
    scale = 0.41
    # 计算相机位置
    eye = [0, 0, 0]
    for i in range(0, 3):
        eye[i] = (carlaTcam[0][i] - carlaTveh[0][i]) * scale

    # 计算相机方向和上方向
    pitch = math.radians(carlaTcam[1][0])
    yaw = math.radians(carlaTcam[1][1])
    roll = math.radians(carlaTcam[1][2])
    # 需不需要确定下范围？？？
    cam_direct = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    cam_up = [math.cos(math.pi / 2 + pitch) * math.cos(yaw), math.cos(math.pi / 2 + pitch) * math.sin(yaw),
              math.sin(math.pi / 2 + pitch)]

    # 如果物体也有旋转，则需要调整相机位置和角度，和物体旋转方式一致
    # 先实现最简单的绕Z轴旋转
    p_cam = eye
    p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]
    p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]
    p_l = [p_cam, p_dir, p_up]
    trans_p = []
    for p in p_l:
        if math.sqrt(p[0] ** 2 + p[1] ** 2) == 0:
            cosfi = 0
            sinfi = 0
        else:
            cosfi = p[0] / math.sqrt(p[0] ** 2 + p[1] ** 2)
            sinfi = p[1] / math.sqrt(p[0] ** 2 + p[1] ** 2)
        cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + sinfi * math.sin(math.radians(carlaTveh[1][1]))
        sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - math.sin(math.radians(carlaTveh[1][1])) * cosfi
        trans_p.append([math.sqrt(p[0] ** 2 + p[1] ** 2) * cossum, math.sqrt(p[0] ** 2 + p[1] ** 2) * sinsum, p[2]])

    return trans_p[0], \
        [trans_p[1][0] - trans_p[0][0], trans_p[1][1] - trans_p[0][1], trans_p[1][2] - trans_p[0][2]], \
        [trans_p[2][0] - trans_p[0][0], trans_p[2][1] - trans_p[0][1], trans_p[2][2] - trans_p[0][2]]


########################################################################
############ Wrapper class for the chainer Neural Renderer #############
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################
class NMR(object):
    """
    神经渲染器包装类
    用于封装neural_renderer库的功能
    """
    def __init__(self):
        """
        初始化渲染器
        """
        # 设置渲染器
        renderer = neural_renderer.Renderer(camera_mode='look')
        self.renderer = renderer

    def to_gpu(self, device=0):
        """
        将渲染器移到GPU
        Args:
            device: GPU设备编号
        """
        # self.renderer.to_gpu(device)
        self.cuda_device = device

    def forward_mask(self, vertices, faces):
        """
        渲染掩码
        Args:
            vertices: 顶点坐标 B X N X 3 numpy数组
            faces: 面片索引 B X F X 3 numpy数组
        Returns:
            masks: 掩码图像 B X 256 X 256 numpy数组
        """
        self.faces = torch.autograd.Variable(faces.cuda())
        self.vertices = torch.autograd.Variable(vertices.cuda())

        self.masks = self.renderer.render_silhouettes(self.vertices, self.faces)

        masks = self.masks.data.get()
        return masks

    # def backward_mask(self, grad_masks):
    #     ''' Compute gradient of vertices given mask gradients.
    #     Args:
    #         grad_masks: B X 256 X 256 numpy array
    #     Returns:
    #         grad_vertices: B X N X 3 numpy array
    #     '''
    #     self.masks.grad = chainer.cuda.to_gpu(grad_masks, self.cuda_device)
    #     self.masks.backward()
    #     return self.vertices.grad.get()

    def forward_img(self, vertices, faces, textures):
        """
        渲染图像
        Args:
            vertices: 顶点坐标 B X N X 3 numpy数组
            faces: 面片索引 B X F X 3 numpy数组
            textures: 纹理 B X F X T X T X T X 3 numpy数组
        Returns:
            images: 渲染图像 B X 3 x 256 X 256 numpy数组
        """
        self.faces = faces
        self.vertices = vertices
        self.textures = textures
        self.images, _, _ = self.renderer.render(self.vertices, self.faces, self.textures)
        return self.images

    # def backward_img(self, grad_images):
    #     ''' Compute gradient of vertices given image gradients.
    #     Args:
    #         grad_images: B X 3? X 256 X 256 numpy array
    #     Returns:
    #         grad_vertices: B X N X 3 numpy array
    #         grad_textures: B X F X T X T X T X 3 numpy array
    #     '''
    #     self.images.grad = chainer.cuda.to_gpu(grad_images, self.cuda_device)
    #     self.images.backward()
    #     return self.vertices.grad.get(), self.textures.grad.get()


########################################################################
################# Wrapper class a rendering PythonOp ###################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Render(torch.autograd.Function):
    """
    渲染操作的PyTorch自动微分函数包装类
    """
    # TODO(Shubham): Make sure the outputs/gradients are on the GPU
    def __init__(self, renderer):
        """
        初始化渲染函数
        Args:
            renderer: 渲染器对象
        """
        super(Render, self).__init__()
        self.renderer = renderer

    def forward(self, vertices, faces, textures=None):
        """
        前向传播函数
        Args:
            vertices: 顶点坐标张量
            faces: 面片索引张量
            textures: 纹理张量（可选）
        Returns:
            渲染结果（掩码或图像）
        """
        # B x N x 3
        # 这里翻转y轴以使其与图像坐标系对齐!
        vs = vertices
        vs[:, :, 1] *= -1
        fs = faces
        if textures is None:
            self.mask_only = True
            masks = self.renderer.forward_mask(vs, fs)
            return masks
        else:
            self.mask_only = False
            ts = textures
            imgs = self.renderer.forward_img(vs, fs, ts)
            return imgs

    # def backward(self, grad_out):
    #     g_o = grad_out.cpu().numpy()
    #     if self.mask_only:
    #         grad_verts = self.renderer.backward_mask(g_o)
    #         grad_verts = convert_as(torch.Tensor(grad_verts), grad_out)
    #         grad_tex = None
    #     else:
    #         grad_verts, grad_tex = self.renderer.backward_img(g_o)
    #         grad_verts = convert_as(torch.Tensor(grad_verts), grad_out)
    #         grad_tex = convert_as(torch.Tensor(grad_tex), grad_out)
    #
    #     grad_verts[:, :, 1] *= -1
    #     return grad_verts, None, grad_tex


########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer(torch.nn.Module):
    """
    神经渲染器的PyTorch模块包装类
    这是调用的核心PyTorch函数。
    每个torch NMR都有一个chainer NMR。
    每次迭代只进行一次前向/反向传播。
    """

    def __init__(self, img_size=800):
        """
        初始化神经渲染器
        Args:
            img_size: 渲染图像的尺寸
        """
        super(NeuralRenderer, self).__init__()
        self.renderer = NMR()

        # 渲染设置
        self.renderer.renderer.image_size = img_size

        # 相机设置
        self.renderer.renderer.camera_mode = 'look'
        self.renderer.renderer.viewing_angle = 45
        eye, camera_direction, camera_up = get_params(((-25, 16, 20), (-45, 180, 0)),
                                                      ((-45, 3, 0.8), (0, 0, 0)))  # 测试示例
        self.renderer.renderer.eye = eye
        self.renderer.renderer.camera_direction = camera_direction
        self.renderer.renderer.camera_up = camera_up

        # 光照设置
        self.renderer.renderer.light_intensity_ambient = 0.5
        self.renderer.renderer.light_intensity_directional = 0.5
        self.renderer.renderer.light_color_ambient = [1, 1, 1]  # 白色
        self.renderer.renderer.light_color_directional = [1, 1, 1]  # 白色
        self.renderer.renderer.light_direction = [0, 0, 1]  # 从上到下

        self.renderer.to_gpu()

        self.proj_fn = None
        self.offset_z = 5.

        self.RenderFunc = Render(self.renderer)

    def ambient_light_only(self):
        """
        设置仅为环境光照明
        """
        # 使光照仅为环境光
        self.renderer.renderer.light_intensity_ambient = 1
        self.renderer.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        """
        设置背景颜色
        Args:
            color: 背景颜色
        """
        self.renderer.renderer.background_color = color

    def project_points(self, verts, cams):
        """
        投影点到图像平面
        Args:
            verts: 顶点坐标
            cams: 相机参数
        Returns:
            投影点坐标
        """
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, textures=None):
        """
        前向传播函数
        Args:
            vertices: 顶点坐标
            faces: 面片索引
            textures: 纹理（可选）
        Returns:
            渲染结果
        """
        if textures is not None:
            return self.RenderFunc.forward(vertices, faces, textures)
        else:
            return self.RenderFunc.forward(vertices, faces)


def example():
    """
    示例函数，演示如何使用神经渲染器
    """
    obj_file = 'test.obj'
    data_path = '../data/phy_attack/train/data132.npz'
    img_save_dir = './render_test_res/'

    vertices, faces = neural_renderer.load_obj(obj_file)

    texture_mask = np.zeros((faces.shape[0], 2, 2, 2, 3), 'int8')
    with open('./all_faces.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1;
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)
    print(texture_mask.size())
    mask_renderer = NeuralRenderer()
    faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
    vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
    # Textures
    texture_size = 2
    textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
    textures = torch.from_numpy(textures).cuda(device=0)
    print(textures.size())
    textures = textures * texture_mask

    data = np.load(data_path)
    img = data['img']
    veh_trans = data['veh_trans']
    cam_trans = data['cam_trans']
    eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)
    mask_renderer.renderer.renderer.eye = eye
    mask_renderer.renderer.renderer.camera_direction = camera_direction
    mask_renderer.renderer.renderer.camera_up = camera_up

    imgs_pred = mask_renderer.forward(vertices_var, faces_var, textures)
    im_rendered = imgs_pred.data.cpu().numpy()[0]
    im_rendered = np.transpose(im_rendered, (1, 2, 0))

    print(im_rendered.shape)
    print(np.max(im_rendered), np.max(img))
    scipy.misc.imsave(img_save_dir + 'test_render.png', im_rendered)
    scipy.misc.imsave(img_save_dir + 'test_origin.png', img)
    # scipy.misc.imsave(img_save_dir + 'test_total.png', np.add(img, 255 * im_rendered))


# def parse_npz():
#     obj_file = 'audi_et.obj'
#     data_path = '../data/phy_attack/train/'

#     vertices, faces = neural_renderer.load_obj(obj_file)
#     mask_renderer = NeuralRenderer()
#     faces_var = torch.autograd.Variable(torch.from_numpy(faces[None, :, :]).cuda(device=0))
#     vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
#     # Textures
#     texture_size = 2
#     textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
#     textures = torch.from_numpy(textures).cuda(device=0)

#     names = os.listdir(data_path)
#     ind = 0
#     for name in names:
#         path = data_path + name
#         data = np.load(path)
#         img = data['img']
#         veh_trans = data['veh_trans']
#         cam_trans = data['cam_trans']
#         cam_trans = cam_trans.astype(np.float64)
#         print('before modify')
#         print(veh_trans)
#         print(cam_trans)