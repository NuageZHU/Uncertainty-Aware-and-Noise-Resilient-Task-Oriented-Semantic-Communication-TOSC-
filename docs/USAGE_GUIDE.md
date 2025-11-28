# 使用指南

完整的语义通信实验框架使用说明。

---

## 📦 安装

### 1. 环境要求
- Python 3.8+
- CUDA 支持的 GPU（推荐，CPU 也可以但慢）

### 2. 安装依赖

```powershell
cd "项目目录"
pip install -r requirements.txt
```

主要依赖：
- `torch` - PyTorch 深度学习框架
- `diffusers` - Stable Diffusion VAE
- `open-clip-torch` - OpenCLIP 语义评估
- `pandas`, `matplotlib`, `seaborn` - 数据分析和可视化

---

## 🚀 快速开始

### 方法 1：快速测试（1 分钟）

验证环境是否正常：

```powershell
python experiments/exp_quick_test.py
```

**输出：**
- `results/quick_test_results.csv`
- 终端显示结果表格

---

### 方法 2：实验1 - 量化和噪声影响（10-60 分钟）

研究不同量化位数和噪声强度对语义保真度的影响：

```powershell
python experiments/exp_quantization_noise.py
```

**输出：**
- `results/quantization_noise_results.csv` - 完整数据
- `results/plots/*.png` - 4 张可视化图表

---

### 方法 3：实验2 - 不确定性阈值扫描（15-30 分钟）

研究传输决策阈值 τ 对传输率和语义质量的权衡：

```powershell
python experiments/exp_tau_scan.py
```

**输出：**
- `results/tau_scan_results.csv` - 聚合统计数据
- `results/plots/tau_tradeoff.png` - 传输率 vs 质量曲线
- `results/plots/tau_transmission_rate.png` - 传输率变化曲线

**关键发现：**
- 揭示传输率和语义质量的帕累托前沿
- 帮助确定最优的自适应传输策略

---

## ⚙️ 配置参数

编辑 `experiments/exp_quantization_noise.py` 调整参数：

### 控制图像数量

```python
# 第 59 行
MAX_IMAGES = 20      # 使用 20 张图（快速）
MAX_IMAGES = 50      # 使用 50 张图（平衡）
MAX_IMAGES = None    # 使用全部 150 张（完整，需要 50-60 分钟）
```

### 调整测试参数

```python
# 第 47-48 行
N_BITS_LIST = [2, 4, 6, 8, 12, 16]           # 量化级别
SIGMA_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3] # 噪声级别
```

**减少参数加快速度：**
```python
N_BITS_LIST = [2, 4, 6, 8]        # 4 种量化
SIGMA_LIST = [0.0, 0.1, 0.2]      # 3 种噪声
# 总实验数：20 × 4 × 3 = 240 次（约 5 分钟）
```

---

## 📁 数据准备

### 图像位置

项目默认使用 `data/all_images/` 文件夹中的图像。

**当前数据：**
- `data/all_images/` - 150 张混合图像
- `data/people/` - 50 张人物图像
- `data/scenes/` - 50 张场景图像
- `data/text/` - 50 张文字图像

### 添加自己的图像

```powershell
# 复制图像到 all_images 文件夹
Copy-Item 你的图片.jpg data/all_images/
```

支持格式：`.png`, `.jpg`, `.jpeg`, `.bmp`

---

## 📊 理解输出

### CSV 数据文件

`results/quantization_noise_results.csv` 包含：

| 列名 | 说明 | 好的范围 |
|------|------|---------|
| `img_name` | 图像文件名 | - |
| `n_bits` | 量化比特数 | 6-8 推荐 |
| `sigma` | 噪声强度 | < 0.15 较好 |
| `sim_local` | 本地重建相似度 | > 0.90 |
| `sim_rx` | 信道后相似度 | > 0.80 可接受 |
| `uncertainty` | 不确定性 | < 0.1 |
| `semantic_degradation` | 语义退化 | < 0.15 |

### 可视化图表

**1. `rate_distortion_quantization.png` - 率失真曲线**
- 展示量化对语义的影响
- 找到"拐点"（性价比最高的比特数）

**2. `noise_robustness.png` - 噪声鲁棒性**
- 对比不同量化方案的抗噪能力
- 线条越平缓 = 越鲁棒

**3. `degradation_heatmap.png` - 语义退化热力图**
- 参数空间的全景视图
- 绿色 = 好，红色 = 差

**4. `3d_rate_noise_distortion.png` - 3D 可视化**
- 三维参数空间
- 山峰 = 好的配置，山谷 = 差的配置

详细分析见：[结果分析文档](./RESULTS_ANALYSIS.md)

---

## 🔧 故障排除

### 问题 1：找不到模块

```
ModuleNotFoundError: No module named 'xxx'
```

**解决：**
```powershell
pip install xxx
# 或重新安装所有依赖
pip install -r requirements.txt
```

---

### 问题 2：CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案 A：** 使用 CPU
```python
# 编辑脚本，找到 DEVICE = ... 那行，改成：
DEVICE = "cpu"
```

**解决方案 B：** 减少图像数量
```python
MAX_IMAGES = 10  # 使用更少图像
```

---

### 问题 3：模型下载慢

首次运行会下载模型（~700MB），需要 5-10 分钟。

**进度检查：**
```powershell
# 查看缓存目录
dir $env:USERPROFILE\.cache\huggingface\hub
```

**加速方法：**
- 使用国内镜像源
- 或耐心等待（只需下载一次）

---

### 问题 4：找不到图像

```
⚠️ No images found in data/all_images
```

**解决：**
```powershell
# 确保在正确的目录
cd "项目根目录"

# 检查图像是否存在
dir data\all_images\*.jpg
dir data\all_images\*.png

# 如果没有，复制一些图像进去
Copy-Item 图片路径 data\all_images\
```

---

### 问题 5：VS Code 黄色警告

导入语句显示黄色波浪线，但程序能运行。

**原因：** VS Code 的 Python 解释器配置不对

**解决（可选）：**
1. 点击右下角 Python 版本
2. 选择正确的解释器（与运行时相同的 Python）

**或直接忽略：** 不影响程序运行

---

## 🎯 实验工作流

### 推荐的实验顺序

#### 第 1 天：环境验证
```powershell
# 1. 快速测试
python experiments/exp_quick_test.py

# 2. 小规模实验（20 张图）
# 编辑 exp_quantization_noise.py: MAX_IMAGES = 20
python experiments/exp_quantization_noise.py
```

#### 第 2 天：完整实验
```powershell
# 使用全部 150 张图
# 编辑 exp_quantization_noise.py: MAX_IMAGES = None
python experiments/exp_quantization_noise.py
```

#### 第 3 天：分析与报告
- 查看生成的图表
- 阅读 `docs/RESULTS_ANALYSIS.md`
- 撰写实验报告

---

## 📚 更多资源

- **实验脚本说明**: `experiments/README.md`
- **结果分析**: `docs/RESULTS_ANALYSIS.md`
- **项目主页**: `README.md`

---

## 💡 常见使用场景

### 场景 1：快速演示（5 分钟）

```powershell
# 使用默认设置运行快速测试
python experiments/exp_quick_test.py

# 查看结果
type results\quick_test_results.csv
```

---

### 场景 2：撰写实验报告（1 小时）

```powershell
# 1. 运行完整实验（20-50 张图）
python experiments/exp_quantization_noise.py

# 2. 打开图表文件夹
explorer results\plots

# 3. 查看数据
type results\quantization_noise_results.csv | more

# 4. 参考分析文档撰写报告
code docs\RESULTS_ANALYSIS.md
```

---

### 场景 3：研究特定参数（15 分钟）

想测试特定的几个配置？

```python
# 编辑 exp_quantization_noise.py
N_BITS_LIST = [6]           # 只测试 6-bit
SIGMA_LIST = [0.0, 0.1, 0.2]  # 三种噪声
MAX_IMAGES = 10             # 10 张图快速测试

# 运行
python experiments/exp_quantization_noise.py
```

---

## 🆘 获取帮助

遇到问题？检查：

1. ✅ Python 版本 >= 3.8
2. ✅ 所有依赖已安装
3. ✅ 图像文件存在且格式正确
4. ✅ 有足够的磁盘空间（至少 2GB）

如果还是有问题，查看：
- 终端的错误信息
- `experiments/README.md` 实验详细说明
- GitHub Issues（如果是开源项目）

---

**祝实验顺利！** 🚀
