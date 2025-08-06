import os
import sys
from PIL import Image
import numpy as np
import tqdm
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")

import nmr_test as nmr
import neural_renderer
from YoloV3.yolo_CAM import YOLO

from data_loader import MyDataset
from torch.utils.data import Dataset, DataLoader
from grad_cam import CAM


from functools import reduce
import argparse



import os
import sys
from PIL import Image
import numpy as np
import tqdm
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")

# sys.path.append("../renderer/")

import nmr_test as nmr
import neural_renderer

from torchvision.transforms import Resize
from data_loader import MyDataset
from torch.utils.data import Dataset, DataLoader
from grad_cam import CAM

import torch.nn.functional as F
import random
from functools import reduce
import argparse
# 设置随机种子以确保实验可重现性
torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)


# 解析命令行参数
parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=5)                 # 训练轮数
parser.add_argument("--lr", type=float, default=0.01)               # 学习率
parser.add_argument("--batchsize", type=int, default=1)             # 批次大小
parser.add_argument("--lamb", type=float, default=1e-4)             # 内容损失的系数lambda
parser.add_argument("--d1", type=float, default=0.9)                # 对边缘区域的保护系数
parser.add_argument("--d2", type=float, default=0.1)                # 对非边缘区域的保护系数
parser.add_argument("--t", type=float, default=0.0001)              # 平滑损失的系数

parser.add_argument("--obj", type=str, default='audi_et_te.obj')    # 3D模型文件路径
parser.add_argument("--faces", type=str, default='./all_faces.txt') # 可训练面片列表文件路径
parser.add_argument("--datapath", type=str, default=r'E:\wmj\DAS\src\data') # 数据集路径
parser.add_argument("--content", type=str, default=r'E:\wmj\DAS\src\textures\smile.npy') # 内容纹理路径
parser.add_argument("--canny", type=str, default=r'E:\wmj\DAS\src\textures\smile_canny.npy') # 边缘mask纹理路径

args = parser.parse_args()


obj_file =args.obj
texture_size = 6
# 加载3D模型文件，包括顶点、面片和纹理信息
vertices, faces, textures = neural_renderer.load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)


mask_dir = os.path.join(args.datapath, 'masks/')


# 启用异常检测，有助于调试
torch.autograd.set_detect_anomaly(True)

log_dir = ""

def make_log_dir(logs):
    """
    创建日志目录，用于存储训练过程中的输出文件
    Args:
        logs: 包含训练参数的字典
    """
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs/' + dir_name
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name



# 从命令行参数中提取训练超参数
T = args.t          # 平滑损失系数
D1 = args.d1        # 边缘区域保护系数
D2 = args.d2        # 非边缘区域保护系数
lamb = args.lamb    # 内容损失系数
LR = args.lr        # 学习率
BATCH_SIZE = args.batchsize  # 批次大小
EPOCH = args.epoch  # 训练轮数


# 加载内容纹理和边缘mask纹理
texture_content = torch.from_numpy(np.load(args.content)).cuda(device=0)
texture_canny = torch.from_numpy(np.load(args.canny)).cuda(device=0)
texture_canny = (texture_canny >= 1).int()

def loss_content_diff(tex):
    """
    计算内容差异损失函数
    该损失函数用于保持纹理在训练过程中尽可能接近原始内容纹理
    Args:
        tex: 当前纹理参数
    Returns:
        内容差异损失值
    """
    return  D1 * torch.sum(texture_canny * torch.pow(tex - texture_content, 2)) + D2 * torch.sum((1 - texture_canny) * torch.pow(tex - texture_content, 2)) 

def loss_smooth(img, mask):
    """
    计算平滑损失函数
    该损失函数用于保持纹理的平滑性，避免出现过于突兀的变化
    Args:
        img: 图像张量
        mask: 掩码张量
    Returns:
        平滑损失值
    """
    # 计算相邻像素之间的差异
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    mask = mask[:, :-1, :-1]
    mask = mask.unsqueeze(1)
    return T * torch.sum(mask * (s1 + s2))

cam_edge = 19

# 用于密度损失计算的全局变量
vis = np.zeros((cam_edge, cam_edge))

def dfs(x1, x, y, points):
    """
    深度优先搜索函数，用于计算连通区域
    Args:
        x1: 输入矩阵
        x: 当前x坐标
        y: 当前y坐标
        points: 存储连通点的列表
    Returns:
        连通区域的点数
    """
    points.append(x1[x][y])
    global vis
    vis[x][y] = 1
    n = 1
    # print(x, y)
    # 向四个方向继续搜索
    if x+1 < cam_edge and x1[x+1][y] > 0 and not  vis[x+1][y]:
        n += dfs(x1, x+1, y, points)
    if x-1 >= 0 and x1[x-1][y] > 0 and not  vis[x-1][y]:
        n += dfs(x1, x-1, y, points)
    if y+1 < cam_edge and x1[x][y+1] > 0 and not  vis[x][y+1]:
        n += dfs(x1, x, y+1, points)
    if y-1 >= 0 and x1[x][y-1] > 0 and not  vis[x][y-1]:
        n += dfs(x1, x, y-1, points)
    return n
        
    
def loss_midu(x1):
    """
    计算密度损失函数
    该损失函数用于鼓励纹理在注意力区域形成连通的块状结构
    Args:
        x1: 注意力图
    Returns:
        密度损失值
    """
    # print(torch.gt(x1, torch.ones_like(x1) * 0.1).float())
    
    x1 = torch.tanh(x1)
    global vis
    vis = np.zeros((cam_edge, cam_edge))
    
    loss = []
    # print(x1)
    # 遍历注意力图中的每个点
    for i in range(cam_edge):
        for j in range(cam_edge):
            # 如果该点有注意力值且未被访问过
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                # 使用DFS找到连通区域
                n = dfs(x1, i, j, point)
                # 计算该连通区域的密度损失
                loss.append( reduce(lambda x, y: x + y, point) / (cam_edge * cam_edge + 1 - n) )
    # print(vis)
    if len(loss) == 0:
        return torch.zeros(1).cuda()
    return reduce(lambda x, y: x + y, loss) / len(loss)

# 初始化纹理参数
# Textures

texture_param = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32') * -0.9# test 0
texture_param = torch.autograd.Variable(torch.from_numpy(texture_param).cuda(device=0), requires_grad=True)

# 加载原始纹理
texture_origin = textures[None, :, :, :, :, :].cuda(device=0)

# 创建纹理mask，标识哪些面片需要被训练
texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
with open(args.faces, 'r') as f:
    face_ids = f.readlines()
    # print(face_ids)
    for face_id in face_ids:
        if face_id != '\n':
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)


def cal_texture(CONTENT=False):
    """
    计算最终纹理
    根据训练参数和原始纹理计算应用mask后的最终纹理
    Args:
        CONTENT: 是否使用内容纹理
    Returns:
        最终纹理张量
    """
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1) 
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    # return textures
    return texture_origin * (1 - texture_mask) + texture_mask * textures
   
         
def run_cam(data_dir, epoch, train=True, batch_size=BATCH_SIZE):
    """
    运行CAM训练过程
    Args:
        data_dir: 数据目录路径
        epoch: 训练轮数
        train: 是否为训练模式
        batch_size: 批次大小
    """
    print(data_dir)
    # 创建数据集和数据加载器
    dataset = MyDataset(data_dir, 608, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,     
        batch_size=batch_size,  
        shuffle=False
    )
    
    # 创建优化器
    optim = torch.optim.Adam([texture_param], lr = LR)
    dnet = YOLO()
    Cam = CAM()
    # 计算初始纹理并设置到数据集中
    textures = cal_texture()

    dataset.set_textures(textures)

    # 开始训练循环
    for _ in range(epoch):
        print('Epoch: ', _, '/', epoch)
        tqdm_loader = tqdm.tqdm(loader)
        # 遍历数据加载器中的每个批次
        for i, (index, total_img, texture_img, masks) in enumerate(tqdm_loader):
            # 保存中间结果图像用于调试
            total_img_np = total_img.data.cpu().numpy()[0]
            total_img_np = Image.fromarray(np.transpose(total_img_np, (1,2,0)).astype('uint8'))

            total_img_np.save(os.path.join(log_dir, 'test_total.jpg')) 

            Image.fromarray((255 * texture_img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(os.path.join(log_dir, 'texture2.png'))
            Image.fromarray((255 * masks).data.cpu().numpy()[0].astype('uint8')).save(os.path.join(log_dir, 'mask.png'))
            # scipy.misc.imsave(os.path.join(log_dir, 'mask.png'), (255*masks).data.cpu().numpy()[0])

            #######

            # 计算CAM注意力图
            #######
            # pred = 0
            #
            # mask, pred = Cam(total_img, index, log_dir)

            inputs = total_img
            mask, pred= dnet.multi_attention(inputs,log_dir, retain_graph=True, use_augmentation=False)
                
            
            ###########
            #   LOSS  #
            ###########
            
            # 计算总损失
            loss = loss_midu(mask) + lamb * loss_content_diff(texture_param) + loss_smooth(texture_img, masks)
            
            
            with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
            
                tqdm_loader.set_description('Loss %.8f, Prob %.8f' % (loss.data.cpu().numpy(), pred))
                f.write('Loss %.8f, Prob %.8f\n' % (loss.data.cpu().numpy(), pred))
                

            ############
            # backward #
            ############
            # 反向传播和参数更新
            if train and loss != 0:
                optim.zero_grad()
                loss.backward(retain_graph=True)
                optim.step()
            # 更新纹理并设置到数据集中
            textures = cal_texture()
            dataset.set_textures(textures)

            
if __name__ == '__main__':
    """
    主函数，程序入口点
    """
    
    # 构建日志参数字典
    logs = {
        'epoch': EPOCH,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'model': 'resnet50',
        'loss_func': 'loss_midu+loss_content+loss_smooth',
        'lamb': lamb,
        'D1': D1,
        'D2': D2,
        'T': T,  
    }
    
    # 创建日志目录
    make_log_dir(logs)
    
    # 构建训练和测试数据目录路径
    train_dir = os.path.join(args.datapath, 'phy_attack\\train\\images')
    test_dir = os.path.join(args.datapath, 'phy_attack\\test\images')

    # 重新加载内容纹理作为初始纹理参数
    texture_param = torch.autograd.Variable(torch.from_numpy(np.load(args.content)).cuda(device=0), requires_grad=True)

    # 运行CAM训练
    run_cam(train_dir, EPOCH)
    
    # 保存训练后的纹理参数
    np.save(os.path.join(log_dir, 'texture.npy'), texture_param.data.cpu().numpy())
    

        
   