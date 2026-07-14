# Reproduction-of-CuDi

本仓库复现CuDi（Curve Distillation），一种继承于Zero-DCE，用于实现高效且可控的图像曝光调整的方法。

## 运行方式

首先配置环境：

```bash
# 创建并且激活环境
conda create -n cudi python=3.10 -y
conda activate cudi

# 安装pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 安装当前项目
pip install -e .
```

训练数据放在目录`data/train`下面，单张图像（无需配对图像）。

下面运行以下脚本，训练教师模型，模型会自动保存在路径`checkpoints/teacher.pt`下。

```bash
python scripts/train_teacher.py
```

