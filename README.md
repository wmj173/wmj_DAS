# DAS: Dual Attention Suppression Attack

DAS (Dual Attention Suppression Attack) 是一种基于双注意力机制的物理对抗攻击方法，旨在通过抑制模型注意力和人类注意力来提升对抗攻击的效果和自然性。

## 项目简介

本项目实现了一种针对目标检测模型（特别是YOLOv3）的物理对抗攻击方法。通过结合模型注意力机制（Grad-CAM/Grad-CAM++）和人类视觉注意力，生成能够有效欺骗目标检测模型的对抗纹理。

主要特点：
- 利用Grad-CAM/Grad-CAM++计算模型注意力
- 通过神经渲染器将对抗纹理映射到3D模型
- 采用双注意力抑制策略提升攻击效果
- 支持多种深度学习模型的测试评估

## 目录结构

```
DAS/
├── src/                    # 源代码目录
│   ├── data_loader.py      # 数据加载器
│   ├── train.py            # 训练脚本
│   ├── test.py             # 测试脚本
│   ├── grad_cam.py         # Grad-CAM实现
│   ├── nmr_test.py         # 神经渲染器
│   ├── YoloV3/             # YOLOv3相关代码
│   ├── models/             # 深度学习模型配置
│   └── yolo_utils/         # YOLO工具函数
├── README.md              # 项目说明文档
└── requirements.txt       # 依赖包列表
```

## 环境依赖

### 硬件要求
- NVIDIA GPU (推荐GTX 1080Ti或更高)
- 至少8GB显存

### 软件依赖
- Python 3.x
- PyTorch 1.4.0
- neural_renderer 1.1.3
- OpenCV
- PIL
- numpy
- tqdm

### 安装步骤

1. 克隆项目代码：
```bash
git clone https://github.com/wmj/DAS.git
cd DAS
```

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 数据准备

在开始训练或测试之前，需要准备以下数据：

1. **数据集**：包含CARLA仿真环境生成的训练和测试数据
2. **3D模型文件**：目标物体的.obj文件和对应的.mtl纹理文件
3. **可训练面片列表**：指定哪些面片需要被训练的txt文件
4. **内容纹理和边缘mask**：初始内容纹理以及对应的边缘mask纹理

## 使用方法

### 训练

使用[train.py](file:///E:/wmj/DAS/src/train.py)脚本进行对抗纹理训练：

```bash
python train.py [参数]
```

#### 训练参数

##### 超参数
- `--epoch`：训练轮数 (默认: 2)
- `--lr`：学习率 (默认: 0.01)
- `--batchsize`：批次大小 (默认: 1)
- `--lamb`：内容损失系数lambda (默认: 1e-4)
- `--d1`：边缘区域保护系数 (默认: 0.9)
- `--d2`：非边缘区域保护系数 (默认: 0.1)
- `--t`：平滑损失系数 (默认: 0.0001)

##### 路径参数
- `--obj`：3D模型文件路径 (默认: 'audi_et_te.obj')
- `--faces`：可训练面片列表文件路径 (默认: './all_faces.txt')
- `--datapath`：数据集路径 (默认: r'E:\wmj\DAS\src\data')
- `--content`：内容纹理路径 (默认: r'E:\wmj\DAS\src\textures\smile.npy')
- `--canny`：边缘mask纹理路径 (默认: r'E:\wmj\DAS\src\textures\smile_canny.npy')

训练结果将存储在`logs/`目录下的相应文件夹中，包括：
- 训练过程中的图像输出 (cam.jpg, cam_b.jpg, cam_p.jpg, mask.png, test_total.jpg, texture2.png)
- loss.txt：训练过程中的loss变化
- texture.npy：训练好的纹理文件

### 测试

使用[test.py](file:///E:/wmj/DAS/src/test.py)脚本进行模型测试：

```bash
python test.py --texture=path_to_texture
```

测试结果将存储在项目根目录下的`acc.txt`文件中，包含使用不同模型（ResNet152、DenseNet201、VGG19、InceptionV3）测试的准确率。

## 核心组件

### Grad-CAM/Grad-CAM++

[grad_cam.py](file:///E:/wmj/DAS/src/grad_cam.py)实现了Grad-CAM和Grad-CAM++算法，用于计算模型注意力图。

### 神经渲染器

[nmr_test.py](file:///E:/wmj/DAS/src/nmr_test.py)基于neural_renderer库，用于将纹理映射到3D模型并渲染到图像中。

### 数据加载器

[data_loader.py](file:///E:/wmj/DAS/src/data_loader.py)负责加载和处理CARLA仿真环境生成的数据。

## 技术细节

### 双注意力机制

1. **模型注意力**：使用Grad-CAM/Grad-CAM++计算目标检测模型的注意力区域
2. **人类注意力**：通过边缘检测等方法模拟人类视觉注意力
3. **注意力抑制**：通过优化目标使生成的对抗纹理在注意力区域尽可能不显眼

### 损失函数

训练过程使用多种损失函数的组合：
1. **密度损失**：鼓励纹理在注意力区域形成连通的块状结构
2. **内容差异损失**：保持纹理在训练过程中尽可能接近原始内容纹理
3. **平滑损失**：保持纹理的平滑性，避免出现过于突兀的变化

## 实验结果

项目通过在多种深度学习模型上测试对抗纹理的有效性，评估攻击效果。

## 相关论文

如果使用本项目，请引用相关论文：

```
@inproceedings{xxx,
  title={DAS: Dual Attention Suppression Attack},
  author={xxx},
  booktitle={xxx},
  year={xxx}
}
```

## 许可证

本项目基于MIT许可证发布，详情请见[LICENSE](file:///E:/wmj/DAS/LICENSE)文件。

## 联系方式

如有任何问题，请联系项目维护者。