1![以下是 **智慧工业防护-高质量合规检测数据集** 的结构化表格介绍：](https://i-blog.csdnimg.cn/direct/5ccead27afc749f6b5e46b93aea847d8.png)


| 属性 | 详情 |
|------|------|
| **数据集名称** | 智慧工业防护-高质量合规检测数据集 |
| **总规模** | 14 GB |
| **图像数量** | 未明确给出，但基于标注实例推算为数千至上万级（含高分辨率图像） |
| **图像分辨率范围** | 最小：1920×1080（Full HD）<br>最大：8192×5462（超高清，接近8K） |
| **标注格式** | 同时提供 **PASCAL VOC（.xml）** 和 **YOLO（.txt）** 两种格式 |
| **总标注实例数** | 超过 **70,000 个** 精细标注目标 |
| **类别数量** | 17 类 |
| **具体类别列表** | 0. 人（Person）<br>1. 头（Head）<br>2. 脸（Face）<br>3. 眼镜（Glasses）<br>4. 医用口罩（Medical Mask）<br>5. 面罩（Face Shield）<br>6. 耳朵（Ear）<br>7. 耳罩（Ear Muffs）<br>8. 手部（Hand）<br>9. 手套（Gloves）<br>10. 脚（Foot）<br>11. 鞋（Shoes）<br>12. 安全马甲（Safety Vest）<br>13. 工具（Tool）<br>14. 安全帽（Safety Helmet）<br>15. 医用服装（Medical Clothing）<br>16. 安全服（Safety Suit） |
| **应用场景** | - 工业安全生产监控<br>- 个人防护装备（PPE）合规检测<br>- 智慧工厂/工地AI巡检<br>- 医疗或洁净车间人员着装规范识别 |
| **适用模型** | YOLOv5/v8/v10、Faster R-CNN、SSD 等主流目标检测框架 |
| **数据特点** | - 高分辨率利于小部件（如耳朵、眼镜）检测<br>- 覆盖人体多部位与防护装备细粒度类别<br>- 支持“是否佩戴”、“是否合规”等高层逻辑判断 |

> 💡 **典型任务示例**：  
> - 检测工人是否佩戴安全帽 + 安全马甲  
> - 判断医护人员是否穿戴医用口罩 + 医用服装  
> - 监测操作人员是否佩戴手套/耳罩等防护用具


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/40509b826268458783b5f7fe36763bf1.png)

1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de19884fa34649f6b25598422352a3fa.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b0ec2fba900b4c7191856a0b87ea781e.png)
针对 **智慧工业防护合规检测数据集**（17类、高分辨率、VOC/YOLO双格式、7万+标注实例）的完整 **YOLOv8 训练方案**，包含：

- ✅ 数据集准备（YOLO格式）
- ✅ 高分辨率训练优化策略
- ✅ 完整可运行的 YOLOv8 训练代码（Python）
- ✅ 自动适配 GPU / 多尺度训练 / 小目标增强
- ✅ 模型评估与推理示例

---

## 📊 一、类别映射（用于 `data.yaml`）

| ID | 类别名称（英文建议命名） |
|----|--------------------------|
| 0 | person |
| 1 | head |
| 2 | face |
| 3 | glasses |
| 4 | medical_mask |
| 5 | face_shield |
| 6 | ear |
| 7 | ear_muffs |
| 8 | hand |
| 9 | gloves |
|10 | foot |
|11 | shoes |
|12 | safety_vest |
|13 | tool |
|14 | safety_helmet |
|15 | medical_clothing |
|16 | safety_suit |

> ⚠️ 注意：若原始标签使用中文或不同命名，请统一转换为上述英文名。

---

## 📁 二、目录结构（YOLO 格式）

```
industrial_ppe_dataset/
├── images/
│   ├── train/      # 训练图像
│   └── val/        # 验证图像
├── labels/
│   ├── train/      # 对应 .txt 标注
│   └── val/
└── industrial_ppe.yaml
```

### `industrial_ppe.yaml` 内容：
```yaml
path: ./industrial_ppe_dataset
train: images/train
val: images/val

nc: 17
names: [
  'person', 'head', 'face', 'glasses', 'medical_mask', 'face_shield',
  'ear', 'ear_muffs', 'hand', 'gloves', 'foot', 'shoes',
  'safety_vest', 'tool', 'safety_helmet', 'medical_clothing', 'safety_suit'
]
```

---

## 🧰 三、环境配置

```bash
# 创建虚拟环境
python -m venv yolov8_ppe
source yolov8_ppe/bin/activate  # Linux/macOS

# 安装依赖
pip install ultralytics opencv-python numpy pandas tqdm
```

> ✅ 支持 CUDA 的 GPU 会自动启用；CPU 模式也可运行（速度较慢）。

---

## 🔧 四、训练优化策略（针对高分辨率 & 小部件）

| 挑战 | 解决方案 |
|------|--------|
| **图像分辨率高（最高 8192×5462）** | 使用 `imgsz=1280` 或 `1536`（平衡显存与细节保留） |
| **小目标多**（如耳朵、眼镜、手套） | 启用 `copy_paste` + `mosaic` 增强 |
| **类别细粒度**（head/face/ear 分离） | 使用更强模型（YOLOv8m 或 YOLOv8l） |
| **遮挡严重**（安全服覆盖身体） | 启用随机裁剪、透视变换增强 |
| **长尾分布**（如 "tool" 较少） | 开启 `copy_paste` 提升稀有类召回率 |

---

## 🚀 五、完整 YOLOv8 训练代码（Python）

```python
# train_industrial_ppe.py
from ultralytics import YOLO
import os
import torch

# 自动选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 使用设备: {device}")

# 1. 加载预训练模型（推荐 YOLOv8m，兼顾精度与小目标）
model = YOLO('yolov8m.pt')  # 可替换为 'yolov8l.pt' 获取更高精度

# 2. 高分辨率训练配置（针对工业防护场景优化）
results = model.train(
    data='industrial_ppe_dataset/industrial_ppe.yaml',
    epochs=120,
    imgsz=1280,                # 关键！保留高分辨率细节（如耳罩、口罩）
    batch=8,                   # 1280 尺寸下 batch=8 需 ~16–24GB 显存
    name='ppe_yolov8m_1280',
    device=device,
    workers=12,
    patience=30,
    save=True,
    exist_ok=True,

    # 数据增强（强化小目标与遮挡鲁棒性）
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.5,                 # 提升低光照场景适应性
    degrees=10.0,
    translate=0.15,
    scale=0.7,                 # 允许更大缩放，生成更小目标
    shear=2.5,
    perspective=0.002,
    fliplr=0.5,
    mosaic=1.0,                # 强烈推荐！提升密集小目标多样性
    mixup=0.1,
    copy_paste=0.3             # 对稀有部件（如 ear, tool）极有效
)

print("✅ 训练完成！最佳模型路径：", os.path.join(results.save_dir, 'weights/best.pt'))
```

> 💡 **显存不足？**
> - 降低 `imgsz=960`
> - 减小 `batch=4`
> - 关闭 `copy_paste`
> - 使用 `yolov8s.pt`

---

## 🔍 六、模型评估与推理示例

### 1. 评估 mAP
```python
from ultralytics import YOLO

model = YOLO('runs/detect/ppe_yolov8m_1280/weights/best.pt')
metrics = model.val(data='industrial_ppe_dataset/industrial_ppe.yaml', imgsz=1280)
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
```

### 2. 单图推理（可视化）
```python
results = model('test_image.jpg', conf=0.3)
annotated_img = results[0].plot()
cv2.imwrite('ppe_result.jpg', annotated_img)
```

### 3. 合规检查逻辑示例（伪代码）
```python
# 示例：检查是否佩戴安全帽 + 安全马甲
boxes = results[0].boxes
classes = [int(cls) for cls in boxes.cls]
if 14 not in classes:
    print("⚠️ 警告：未检测到安全帽！")
if 12 not in classes:
    print("⚠️ 警告：未检测到安全马甲！")
```

---


实现对 **人体部位 + 防护装备** 的精准识别，适用于 **智慧工厂、建筑工地、医疗车间** 等高安全要求场景。


