# qwen3-vl-2b + videocode(x264) 视频宏块级 QP 分配示例


环境信息：
python:3.10.20
pytorch:2.4.1
pytorch-cuda:2.4.1

## 功能

- 使用 Qwen3-VL-2B（本地部署）：
  1. 观看整个视频（通过抽取关键帧）做全局语义理解；
  2. 对视频的每一帧进行 **16×16 像素宏块划分**，并为每个宏块生成 QP 分配建议；
  3. 生成的 QP 矩阵对接 **videocode.exe（内部使用 x264，16×16 宏块）**；
  4. 可选：Python 直接把每帧 YUV420 + QP 图按二进制协议推送到 `videocode.exe`，完成端到端编码。

## 数据格式

你的数据可以直接按 KITTI 序列目录组织：

- `data/kitti/training/0000/000000.png ...`
- `data/kitti/training/0001/000000.png ...`
- `data/kitti/training/0020/000000.png ...`

脚本默认按 `10 fps` 解释这些图像序列的时间间隔。

## 目录结构

- `main.py`：主入口，支持视频文件和 KITTI 序列目录，生成宏块级 QP JSON，可选调用 x264 编码。
- `video_utils.py`：视频/图像序列读取、关键帧抽取、宏块划分（支持任意 block_size，推荐 16）。
- `qwen_vl_client.py`：封装 Qwen3-VL-2B 本地调用，输出每帧宏块级 QP。
- `x264_runner.py`：读取 `qp_map.json`，将视频文件或图像序列帧打包为二进制协议并通过 stdin 推流给 `videocode.exe`。
- `README.md`：说明文档。

## 使用

输出结构：所有编码结果都保存在 `outputs` 文件夹下：
```
outputs/
  └── base/           # 基线编码（统一 QP）
      ├── 0000.h264
      ├── 0001.h264
      └── ...
  └── qp/             # 自适应 QP 编码（Qwen VL 分配）
      ├── 0000.json   # QP 分配信息
      ├── 0000.h264   # 对应的码流
      ├── 0001.json
      ├── 0001.h264
      └── ...
```

### 1. 快速测试

一帧
<!-- python test_first_sequence.py 
  --input-root E:/AiLearning/vlmvedio/data/kitti/training 
  --output-dir E:/AiLearning/vlmvedio/outputs_test_first 
  --encode-x264 
  --x264-bin E:/AiLearning/vlmvedio/videocode/x64/Debug/videocode.exe 
  --quick-validate -->
  python test_first_sequence.py --input-root E:/AiLearning/vlmvedio/data/kitti/training --output-dir E:/AiLearning/vlmvedio/outputs_test_first --encode-x264 --x264-bin E:/AiLearning/vlmvedio/videocode/x64/Debug/videocode.exe --quick-validate

第一序列
python test_first_sequence.py --input-root E:/AiLearning/vlmvedio/data/kitti/training --output-dir E:/AiLearning/vlmvedio/outputs/qp --encode-x264 --x264-bin E:/AiLearning/vlmvedio/videocode/x64/Debug/videocode.exe




### 2. 自适应 QP 编码（主方法）

2.1 仅生成 QP JSON：

```bash
python main.py \
  --input-root E:/AiLearning/vlmvedio/data/kitti/training \
  --block-size 16 \
  --qp-contrast 1.6
```

输出到 `outputs/qp/`

2.2 生成 QP 并直接编码：

```bash
python main.py \
  --input-root E:/AiLearning/vlmvedio/data/kitti/training \
  --block-size 16 \
  --qp-contrast 1.6 \
  --encode-x264 \
  --x264-bin E:/AiLearning/vlmvedio/videocode/x64/Debug/videocode.exe \
  --fps 10
```

输出到 `outputs/qp/`

### 3. 基线编码（统一 QP，用于对比）

用统一的 QP 值对同一数据集进行编码，不经过 Qwen VL：

```bash
python baseline_encoder.py \
  --input-root E:/AiLearning/vlmvedio/data/kitti/training \
  --qp 28 \
  --x264-bin E:/AiLearning/vlmvedio/videocode/x64/Debug/videocode.exe
```

输出到 `outputs/base/`

单独处理一个序列：

```bash
python baseline_encoder.py \
  --single-sequence E:/AiLearning/vlmvedio/data/kitti/training/0000 \
  --qp 28 \
  --x264-bin E:/AiLearning/vlmvedio/videocode/x64/Debug/videocode.exe
```

输出到 `outputs/base/`

### 4. 对比码率差异

编码完成后，自动生成对比报告：

```bash
python compare_encodings.py \
  --input-root E:/AiLearning/vlmvedio/data/kitti/training
```

输出示例：
```
编码对比报告
序列名               帧数 自适应QP       基线(MB)    差异(%)    QP范围
0000               154 [10.0~30.8]     
  μ=23.5            12.34          12.45         -0.9      10.0~30.8
0001               147 [12.1~31.2]     
  μ=24.1            11.89          12.10         -1.7      12.1~31.2
...
总计              1500                         123.45         -1.4

自适应 QP 方法更优，节省了 1.4%
详细报告已保存到: outputs/qp/comparison_report.json
```

### 5. 旧的单视频模式仍可用

```bash
python main.py --video input.mp4 --output qp_map.json --block-size 16
```

## 参数说明

### main.py - 自适应 QP 编码

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-root` | - | KITTI 序列目录的根路径 |
| `--output-dir` | `outputs/qp` | 输出 JSON/H.264 文件的目录 |
| `--block-size` | 16 | 宏块大小（像素），x264 推荐 16 |
| `--qp-contrast` | 1.0 | QP 差异强度（0.5~3.0，越大差异越明显） |
| `--encode-x264` | - | 生成 QP 后立即调用 videocode.exe 编码 |
| `--x264-bin` | `videocode.exe` | videocode.exe 的路径 |
| `--max-frames` | - | 限制每个序列最多处理的帧数 |
| `--fps` | 10.0 | 帧率（用于解释图像序列的时间间隔） |

### baseline_encoder.py - 基线编码

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-root` | - | KITTI 序列目录或单个序列路径 |
| `--output-dir` | `outputs/base` | 输出 H.264 文件的目录 |
| `--qp` | 28.0 | 统一的量化参数（0~51） |
| `--x264-bin` | `videocode.exe` | videocode.exe 的路径 |
| `--max-frames` | - | 每个序列最多编码的帧数 |
| `--single-sequence` | - | 仅处理指定的单个序列目录 |

### compare_encodings.py - 对比报告

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-root` | - | KITTI 序列目录的根路径 |
| `--adaptive-output` | `outputs/qp` | 自适应 QP 编码的输出目录 |
| `--baseline-output` | `outputs/base` | 基线编码的输出目录 |

## 快速对比工作流

1. **生成自适应 QP 编码**（约 30~60 分钟，取决于序列总帧数）：
   ```bash
   python main.py \
     --input-root data/kitti/training \
     --qp-contrast 1.6 \
     --encode-x264 \
     --x264-bin videocode/x64/Debug/videocode.exe
   ```
   输出到 `outputs/qp/`

2. **生成基线编码**（QP=28）：
   ```bash
   python baseline_encoder.py \
     --input-root data/kitti/training \
     --qp 28 \
     --x264-bin videocode/x64/Debug/videocode.exe
   ```
   输出到 `outputs/base/`

3. **对比码率**：
   ```bash
   python compare_encodings.py \
     --input-root data/kitti/training
   ```
   对比报告自动输出到 `outputs/qp/comparison_report.json`

## Python-C++ 数据契约（A 路径）

Python 端发送的每帧包结构如下（小端）：

1. `magic[4] = "QPK1"`
2. `version(uint32) = 1`
3. `frameId(uint32)`
4. `width(uint32)`
5. `height(uint32)`
6. `qpWidth(uint32)`
7. `qpHeight(uint32)`
8. `ySize(uint32)`
9. `uSize(uint32)`
10. `vSize(uint32)`
11. `qpSize(uint32)`
12. `y_data` + `u_data` + `v_data` + `qp_data`

其中：

- `y/u/v` 是 I420 平面数据。
- `qp_data` 是按行优先压平的 `uint8` 数组，值范围 `[0, 51]`。
- 默认缺帧回退 QP 为 28。

## 构建与运行提示（Windows）

1. 在 Visual Studio 使用 `x64|Debug` 或 `x64|Release` 编译 `videocode` 工程，确认生成 `videocode.exe`。
2. 在 VS Code/Python 环境运行 `main.py`，优先使用 `--input-root` 批量处理序列。
3. 若编码失败，优先检查：
   - `--x264-bin` 路径是否指向真实 `videocode.exe`；
  - 图像序列帧宽高是否能正常读取，脚本会自动补成偶数尺寸；
   - QP JSON 是否包含对应帧号。
