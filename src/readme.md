# Physical Attack

## 项目概述

DAS (Dual Attention Suppression Attack) 是一种基于双注意力机制的物理对抗攻击方法，旨在通过抑制模型注意力和人类注意力来提升对抗攻击的效果和自然性。

### 运行前依赖

- 数据集
- 3D obj 文件，和对应的 mtl纹理文件
- 需要被训练的face的id列表（保存成txt）
- 被训练的face对应的初始内容纹理content，以及内容纹理对应的边缘mask纹理 canny

### 项目结构

```
src/
├── data_loader.py          # 数据加载器
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── grad_cam.py             # Grad-CAM实现
├── nmr_test.py             # 神经渲染器
├── myRandomAffine.py       # 随机仿射变换
├── YoloV3/                 # YOLOv3相关代码
│   ├── attention.py        # 注意力机制实现
│   ├── yolo_CAM.py         # YOLO CAM实现
│   ├── yolov5.py           # YOLOv5模型
│   ├── nets/               # 网络定义
│   ├── utils/              # 工具函数
│   └── misc_functions.py   # 杂项函数
├── models/                 # 深度学习模型配置
├── yolo_utils/             # YOLO工具函数
└── readme.md              # 本文件
```

### 训练

```shell
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

训练结果存储在`logs/`目录下相应的文件夹内，其中包括：

- 训练过程中的一些图片输出
  - cam.jpg    gradcam图
  - cam_b.jpg  CAM二值图
  - cam_p.jpg  纯CAM热力图
  - mask.png   掩码图像
  - test_total.jpg  模型输入图片展示
  - texture2.png    纹理图像
- loss.txt   训练过程中的loss变化
- texture.npy   训练好的纹理文件

### 测试

```shell
python test.py --texture=path_to_texture [--obj=obj_file] [--datapath=data_path]
```

测试参数：
- `--texture`：训练好的纹理文件路径
- `--obj`：3D模型文件路径 (可选)
- `--datapath`：数据集路径 (可选)

测试结果存储在项目根目录下的`new_acc.txt`文件中，包含使用多种模型（ResNet152、DenseNet201、VGG19、InceptionV3）测试的准确率。

### 核心技术

#### 双注意力机制

1. **模型注意力**：使用Grad-CAM/Grad-CAM++计算目标检测模型的注意力区域
2. **人类注意力**：通过边缘检测等方法模拟人类视觉注意力
3. **注意力抑制**：通过优化目标使生成的对抗纹理在注意力区域尽可能不显眼

#### 损失函数

训练过程使用多种损失函数的组合：
1. **密度损失** (`loss_midu`)：鼓励纹理在注意力区域形成连通的块状结构
2. **内容差异损失** (`loss_content_diff`)：保持纹理在训练过程中尽可能接近原始内容纹理
3. **平滑损失** (`loss_smooth`)：保持纹理的平滑性，避免出现过于突兀的变化

#### 神经渲染器

使用`neural_renderer`库将纹理映射到3D模型并渲染到图像中，实现物理世界中的对抗攻击效果。

### 注意事项

1. 项目需要GPU支持，推荐使用NVIDIA显卡
2. 确保数据集路径正确，并包含训练和测试数据
3. 3D模型文件(.obj)和材质文件(.mtl)需要配套使用
4. 训练过程可能需要较长时间，建议根据硬件配置调整训练参数