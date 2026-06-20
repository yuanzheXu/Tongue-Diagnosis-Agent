# Tongue Diagnosis Agent

基于舌象的智能辅助诊断系统：**PIDNet 舌体分割 → YOLOv8 齿痕/舌裂检测 → LLM 综合诊断**。

## 项目结构

```text
Tongue-Diagnosis-Agent/
├── data/
│   ├── fissure/                          # 舌裂原始标注 (LabelMe JSON)
│   │   ├── annotate/                     #   有舌裂: 892 张
│   │   └── label-free/                   #   无舌裂: 992 张
│   ├── teeth indentations on tongue/     # 齿痕原始标注 (LabelMe JSON)
│   │   ├── annotate/                     #   有齿痕: 372 张
│   │   └── label-free/                   #   无齿痕: 1431 张
│   ├── tongue segmentation/              # PIDNet 训练数据 (tongue1, 996张)
│   │   ├── tongue/    (原图)
│   │   ├── mask/      (binary mask, 0/255)
│   │   ├── json/      (LabelMe polygon)
│   │   └── tongue_only/ (masked output)
│   ├── yolo_dataset/                     # YOLOv8 训练数据 (1642张)
│   │   ├── images/          (PIDNet遮罩图: 背景→黑色)
│   │   ├── images_original/ (原图备份, 供LLaVA使用)
│   │   ├── labels/          (YOLO txt, 1142个有标注)
│   │   ├── masked_images/   (遮罩推理缓存)
│   │   ├── masks/           (PIDNet binary masks)
│   │   ├── json/            (原始LabelMe JSON备份)
│   │   └── data.yaml        (YOLO训练配置)
│   └── tongue_agent_dataset/             # LLM 训练数据集
│       ├── tongue_diagnosis_structured.json  (完整结构化, 1640条)
│       ├── tongue_diagnosis_alpaca.json      (Alpaca指令微调)
│       └── tongue_diagnosis_llava.json       (LLaVA多模态)
├── tongue PID segmentation/              # PIDNet 舌体分割
│   ├── tools/train.py                    #   训练入口
│   ├── tools/extract_tongues.py          #   推理脚本
│   ├── tools/output/tongue/pidnet_small_tongue/best.pt  # 已训练权重
│   └── configs/tongue/                   #   配置文件
└── tongue YOLO detection/                # YOLOv8 齿痕/舌裂检测
    ├── scripts/
    │   ├── train_yolo.py                 #   训练入口
    │   ├── pidnet_segment_all.py         #   PIDNet批量推理
    │   ├── prepare_yolo_dataset_v2.py    #   数据集构建
    │   └── build_llm_dataset_v2.py       #   LLM数据集生成
    └── runs/detect/runs/tongue_detect/   #   训练输出
        └── weights/best.pt               #   最佳YOLO模型
```

## 流水线

```text
舌图 → PIDNet分割舌体(遮罩背景) → YOLOv8检测齿痕/舌裂 → +患者病历 → LLM诊断
                                    ↓
                          齿痕个数 + 舌裂个数 + 位置
```

## 模型

| 模型 | 任务 | 性能 | 权重路径 |
|------|------|------|----------|
| PIDNet-S | 舌体分割 | mIoU 0.9883 | `tongue PID segmentation/tools/output/tongue/.../best.pt` |
| YOLOv8s | 齿痕+舌裂检测 | mAP50 0.574 | `tongue YOLO detection/runs/detect/runs/tongue_detect/weights/best.pt` |

## 实验结果展示

YOLOv8s 在掩膜处理后的舌象数据上，对齿痕（teeth_mark）与舌裂（fissure）的检测效果对比：

<p align="center">
  <strong>真实标注 (Ground Truth)</strong><br>
  <img src="tongue%20YOLO%20detection/runs/detect/runs/tongue_detect/val_batch0_labels.jpg" width="800" alt="Validation Batch 0 Labels">
</p>

<p align="center">
  <strong>模型预测 (Predictions)</strong><br>
  <img src="tongue%20YOLO%20detection/runs/detect/runs/tongue_detect/val_batch0_pred.jpg" width="800" alt="Validation Batch 0 Predictions">
</p>

## 快速开始

### 0. 环境准备

推荐使用 Conda 管理环境。请确保您已安装所需依赖：
```bash
conda activate vmorph
# 环境依赖: torch 2.7.1+cu118, ultralytics 8.4.71, CUDA 11.8
```

> **注意：** 以下所有命令均需在 `Tongue-Diagnosis-Agent` 项目根目录下运行。

### 1. PIDNet 分割

对所有 YOLO 数据集图片做舌体分割，输出 mask 和遮罩图：
```bash
python "tongue YOLO detection/scripts/pidnet_segment_all.py"
```

### 2. YOLOv8 训练

在遮罩图上训练齿痕+舌裂检测：
```bash
python "tongue YOLO detection/scripts/train_yolo.py"
```

### 3. 生成 LLM 数据集

匹配舌图编号与患者病历（`全_16000例_去掉姓名和隐私信息.xlsx`），生成三种格式的 LLM 训练数据：
```bash
python "tongue YOLO detection/scripts/build_llm_dataset_v2.py"
```

## 数据集统计

| 项目 | 数值 |
|------|------|
| 总图片数 | 2595 张 (来自 fissure + teeth 数据集) |
| YOLO 训练集 | 1642 张 (1142 有标注 + 500 负样本) |
| 匹配病历 | 1640 条 (Excel 16,102 行中匹配) |
| 舌裂标注框 | 1,360 个 |
| 齿痕标注框 | 1,110 个 |
| YOLO检测-舌裂 | 890 张 (54.3%) |
| YOLO检测-齿痕 | 371 张 (22.6%) |
| 文本描述-齿痕 | 1,579 张 (96.3%) |
| 文本描述-裂纹 | 464 张 (28.3%) |

## LLM 数据集格式

### Alpaca (指令微调)

```json
{
  "instruction": "你是一名中医舌诊专家...",
  "input": "性别：男\n年龄：55岁\nBMI：17.1\nAI舌象检测：检测到舌裂0处，齿痕0处",
  "output": "舌象分析：舌色淡红，苔黄薄，舌有裂纹\n脉象：细脉\n中医体质：平和质\n内科诊断：慢性萎缩性胃炎"
}
```

### LLaVA (多模态)

```json
{
  "id": "1000125596",
  "image": "data/yolo_dataset/images_original/train/1000125596.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n患者：男，55岁\nBMI：17.1\nAI检测：舌裂0处，齿痕0处"},
    {"from": "gpt", "value": "舌象：舌色淡红，苔黄薄，舌有裂纹\n体质：平和质\n诊断：慢性萎缩性胃炎"}
  ]
}
```

## 数据来源

- **舌图 + 齿痕/舌裂标注**: 舌诊数据集 (LabelMe 格式，两批标注合并)
- **舌体轮廓**: `Tongue-contour` (tongue1-tongue7, LabelMe polygon)
- **患者病历**: `全_16000例_去掉姓名和隐私信息.xlsx` (16,102 行 × 443 列)
  - 含舌象诊断、脉象、中医体质、内科诊断、基础体征等
