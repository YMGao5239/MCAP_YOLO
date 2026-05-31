# MCAP YOLO Image Quality Gateway

从机器人采集的 **MCAP** 文件中解析相机图像帧,执行**图像数据质量评估**,
对合格帧运行 **YOLO ONNX** 目标检测与类别识别,并生成 JSON/HTML/Markdown 报告。
通过 **FastAPI + Docker Compose** 一键交付。

> 作业版本 v1.0 · 离线工程作业 · 建议工时 12~18h

---

## 1. 项目简介
评分重点不是检测精度,而是 **「MCAP 图像帧 → 检测结果」完整工程链路** 的正确性与工程化质量。

数据流:
```
MCAP File/Dir → MCAP Reader → Topic Scanner → ROS Image Decoder
  → Per-Topic Frame Iterator → Image Quality Analyzer
  → Qualified Frames → YOLO Detection + Classification
  → Detection Results + Latency → JSON / HTML / Markdown Report
```

## 2. 功能概览
- MCAP 解析(单文件 + 目录批处理,单文件失败不影响其他)
- 自动发现 / 手动指定图像 Topic
- `sensor_msgs/msg/CompressedImage` 与 `Image` 解码
- 单帧质量指标 + 质量评分 + 按 Topic 汇总 + 视频时序(FPS)分析
- YOLO ONNX Runtime 推理(自写前后处理 + NMS)+ 关键类别过滤 + 质量门控
- FastAPI 服务 + Docker Compose 一键启动

## 3. MCAP 输入形式说明
支持两种输入形式:

- 单个 MCAP 文件: 通过 `--mcap path/to/file.mcap` 指定。
- 目录批处理: 通过 `--mcap-dir path/to/dir` 递归扫描目录下所有 `.mcap` 文件;单个文件读取失败会进入 `failed_files`,不影响其他文件。

示例:

```bash
python scripts/run_mcap_quality_scan.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --output-dir ./outputs

python scripts/run_mcap_quality_scan.py \
  --mcap-dir ./test_data \
  --auto-detect-topics true \
  --sample-every-n 5 \
  --output-dir ./outputs
```

MCAP 扫描脚本支持抽帧与时间裁剪参数,并在 `mcap_summary.json` 中记录采样配置:

```bash
python scripts/run_mcap_quality_scan.py \
  --mcap ./test_data/sample.mcap \
  --sample-every-n 5 \
  --start-sec 2.0 \
  --end-sec 12.0 \
  --output-dir ./outputs
```

测试 MCAP 可用 `scripts/generate_test_mcap.py` 生成,见第 8 节。

## 4. 支持的 ROS 图像消息类型
当前支持两类 ROS2 图像消息:

| 消息类型 | 处理方式 |
|---|---|
| `sensor_msgs/msg/CompressedImage` | 读取 `format` 与 `data`,使用 `cv2.imdecode` 解码 JPEG/PNG 等压缩图像,输出 OpenCV BGR 图像 |
| `sensor_msgs/msg/Image` | 读取 `height / width / encoding / is_bigendian / step / data`,按 encoding 还原像素并转换为 BGR |

`Image.encoding` 已支持:

```text
bgr8, rgb8, bgra8, rgba8, mono8, 8UC1, 8UC3, 8UC4,
mono16, 16UC1,
bayer_rggb8, bayer_bggr8, bayer_gbrg8, bayer_grbg8
```

未知 encoding、空 payload、损坏压缩帧会被计入解码失败或不支持 encoding 统计,单帧失败不会打断整条管线。

## 5. 环境依赖
运行环境:

- Python 3.10+
- 依赖见 `requirements.txt`
- MCAP 读取使用 `rosbags`
- 图像处理使用 `opencv-python-headless` / `numpy`
- YOLO 推理使用 `onnxruntime` CPUExecutionProvider
- 服务接口使用 `fastapi` / `uvicorn`

本地安装:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ultralytics` 只在重新导出 ONNX 模型时需要,不属于运行时和 Docker 镜像依赖。

## 6. Docker Compose 一键启动
```bash
docker compose up --build
# 访问 http://127.0.0.1:8000/docs
```

## 7. 本地运行方式
进入项目目录后运行:

```bash
cd mcap_yolo_quality_gateway
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

生成测试数据:

```bash
python scripts/generate_test_mcap.py
```

只做 MCAP 元信息与 Topic 扫描:

```bash
python scripts/run_mcap_quality_scan.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --output-dir ./outputs
```

执行质量门控 + YOLO 推理:

```bash
python scripts/run_mcap_yolo_inference.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --model ./models/yolov8n.onnx \
  --labels ./models/coco_classes.txt \
  --target-classes person,car,truck,bus \
  --sample-every-n 5 \
  --quality-threshold 0.6 \
  --conf-threshold 0.25 \
  --nms-threshold 0.45 \
  --output-dir ./outputs
```

启动 API 服务:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 访问 http://127.0.0.1:8000/docs
```

## 8. 生成测试 MCAP
生成一个可复现的合成 MCAP 文件,同时包含 raw `Image` 与 `CompressedImage` 两种图像 Topic:

```bash
python scripts/generate_test_mcap.py
```

默认产物:

```text
test_data/sample.mcap
```

默认 Topic:

```text
/camera/image       sensor_msgs/msg/Image
/camera/compressed  sensor_msgs/msg/CompressedImage
```

样本中会混入正常帧、全黑帧、模糊帧、低分辨率帧,用于后续验证解码、质量评分和坏样本导出。

可选参数:

```bash
python scripts/generate_test_mcap.py --output test_data/sample.mcap --frames 30 --fps 10
```

## 9. 下载与运行 YOLO ONNX 模型
> ⚠️ 这是一次性的**离线/宿主机**步骤,**不在 Docker 镜像内执行**。
> 导出依赖 `ultralytics`(会拖入 torch + 捆绑 CUDA 的 wheel),因此它**不在** `requirements.txt`/镜像里。
> 仓库已附带导出好的 `models/yolov8n.onnx`,验收时无需重新导出;
> 容器内推理只用 `onnxruntime` CPUExecutionProvider。

仅当需要重新生成 ONNX 时,在宿主机单独安装并运行(用 Ultralytics 下载 `yolov8n.pt`,导出 ONNX,并用 ONNX Runtime CPU provider 验证输入/输出张量):

```bash
pip install ultralytics          # 仅导出时需要, 不进镜像
python scripts/download_yolo_model.py
```

默认产物:

```text
models/yolov8n.onnx
```

可选参数:

```bash
python scripts/download_yolo_model.py --model yolov8n.pt --output models/yolov8n.onnx --opset 12 --imgsz 640
```

后续推理必须通过 `onnxruntime.InferenceSession(..., providers=["CPUExecutionProvider"])` 加载该 ONNX 模型。

## 10. 目录批处理 / 数据质量评估
目录批处理入口用于 MCAP 元信息解析与图像 Topic 发现:

```bash
python scripts/run_mcap_quality_scan.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --output-dir ./outputs
```

目录递归扫描:

```bash
python scripts/run_mcap_quality_scan.py \
  --mcap-dir ./test_data \
  --auto-detect-topics true \
  --sample-every-n 5 \
  --output-dir ./outputs
```

主要参数:

| 参数 | 说明 |
|---|---|
| `--mcap` | 单个 MCAP 文件路径 |
| `--mcap-dir` | 递归扫描目录中的 `.mcap` 文件 |
| `--auto-detect-topics` | 自动选择 ROS 图像 Topic |
| `--topics` | 手动指定 Topic 列表 |
| `--sample-every-n` | 每 N 帧采样 1 帧 |
| `--start-sec` / `--end-sec` | 基于 MCAP 起始时间的时间裁剪 |
| `--output-dir` | 输出目录 |

输出:

```text
outputs/mcap_summary.json
```

完整数据质量评估由管线函数 `run_quality_scan()` 执行,当前通过 FastAPI `/mcap/quality_scan` 暴露;YOLO 推理脚本也会先执行同一套质量评估再进入推理阶段,并写出:

```text
outputs/quality_report.json
outputs/quality_report.md
outputs/quality_report.html
outputs/metrics.json
outputs/bad_samples/index.json
```

## 11. 运行 YOLO 推理
YOLO 推理入口:

```bash
python scripts/run_mcap_yolo_inference.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --model ./models/yolov8n.onnx \
  --labels ./models/coco_classes.txt \
  --target-classes person,car,truck,bus \
  --sample-every-n 5 \
  --quality-threshold 0.6 \
  --conf-threshold 0.25 \
  --nms-threshold 0.45 \
  --output-dir ./outputs
```

主要参数:

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `models/yolov8n.onnx` | ONNX 模型路径 |
| `--labels` | `models/coco_classes.txt` | COCO 类别文件 |
| `--target-classes` | 无 | 只保留关心类别,如 `person,car,truck,bus` |
| `--quality-threshold` | `0.6` | 低于阈值标记为 `bad_quality` |
| `--conf-threshold` | `0.25` | 检测置信度阈值 |
| `--nms-threshold` | `0.45` | NMS IoU 阈值 |
| `--infer-low-quality` | `false` | 是否仍对低质量帧运行 YOLO |
| `--max-bad-samples` | `20` | 最多导出低质量样本数 |
| `--max-detection-samples` | `20` | 最多导出检测可视化样本数 |
| `--max-frames` | 无 | 限制最多处理帧数,便于 smoke test |

额外输出:

```text
outputs/yolo_predictions.json
outputs/detection_samples/index.json
```

## 12. FastAPI 接口说明
启动后访问:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/docs
```

接口概览:

| 接口 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 服务入口 |
| `/health` | GET | 健康检查,返回模型文件状态、MCAP reader、ONNX Runtime 版本 |
| `/mcap/inspect` | POST | 解析 MCAP 元信息并自动发现图像 Topic |
| `/mcap/quality_scan` | POST | 执行质量扫描并写出报告 |
| `/mcap/yolo_infer` | POST | 执行质量门控 + YOLO 推理并写出预测 |
| `/jobs/{job_id}` | GET | 查询任务状态 |
| `/mcap/frame` | GET | 返回指定帧 JPEG base64 预览 |
| `/mcap/frame_yolo` | GET | 返回指定帧 YOLO 可视化 JPEG base64 预览 |

`/mcap/quality_scan` 请求示例:

```json
{
  "mcap_path": "test_data/sample.mcap",
  "output_dir": "outputs",
  "auto_detect_topics": true,
  "sample_every_n": 5,
  "quality_threshold": 0.6,
  "max_frames": 20
}
```

`/mcap/yolo_infer` 请求示例:

```json
{
  "mcap_path": "test_data/sample.mcap",
  "output_dir": "outputs",
  "auto_detect_topics": true,
  "model_path": "models/yolov8n.onnx",
  "labels_path": "models/coco_classes.txt",
  "target_classes": "person,car,truck,bus",
  "sample_every_n": 5,
  "quality_threshold": 0.6,
  "conf_threshold": 0.25,
  "nms_threshold": 0.45,
  "infer_low_quality": false
}
```

任务接口当前在请求内同步执行管线,完成后返回 `job_id / status / result_path / report_path`。

## 13. 质量评分规则
每帧先计算可解释质量指标,再按扣分项得到 `quality_score`:

```text
score = 1.0
  - blur_penalty
  - exposure_penalty
  - contrast_penalty
  - resolution_penalty
  - corruption_penalty
  - timestamp_penalty
```

默认阈值为 `--quality-threshold 0.6`,低于阈值追加 `bad_quality` 标签。

默认规则:

| 问题 | 判定 |
|---|---|
| 过暗 | `brightness_mean <= 35` |
| 过亮 | `brightness_mean >= 245` |
| 模糊 | `blur_score < 80`,空帧不重复标模糊 |
| 低对比度 | `contrast_score < 18` |
| 低分辨率 | `width < 320` 或 `height < 240` |
| 损坏/空帧 | 空帧、非法 ndarray 或指标计算失败 |

输出会保留 `quality_score`、`quality_tags`、`quality_flags`、`penalties` 和原始指标,便于追溯每个坏样本原因。详细规则见 `docs/quality_rules.md`。

## 14. 质量门控说明
质量门控位于 YOLO 推理前:

1. 解码图像帧。
2. 计算质量指标与质量分。
3. 当 `quality_score < quality_threshold` 时标记为 `bad_quality`。
4. 默认 `--infer-low-quality false`,低质量帧跳过 YOLO 推理并计入 `skipped_low_quality_frames`。
5. 如需排查模型在坏帧上的行为,可设置 `--infer-low-quality true` 强制推理。

这样可以避免在全黑、模糊、损坏或低分辨率帧上浪费 CPU 推理时间,同时报告仍会保留低质量帧的质量分析和坏样本导出。

## 15. 时间序列 / FPS 分析
按 Topic 收集帧时间戳,输出在 `quality_report.json` 的 `sequence` 字段中:

```text
estimated_fps
frame_interval_ms_avg
frame_interval_ms_p95
timestamp_jump_count
long_gap_count
```

计算方式:

- 使用相邻帧 `log_time_ns` 差值估计帧间隔。
- `estimated_fps = 1000 / median_interval_ms`。
- 非递增时间戳计入 `timestamp_jump_count`。
- 明显长间隔计入 `long_gap_count`,默认以中位间隔的 3 倍作为参考。

这些指标用于发现掉帧、时间戳跳变、采集频率异常等视频时序问题。

## 16. 分类与类别说明
本项目中的 YOLO 输出同时包含检测与类别:

- 检测: 定位每个目标的 `bbox_xyxy`。
- 分类: 对每个检测框输出 `class_id / label / confidence`。

默认类别文件为 COCO 80 类:

```text
models/coco_classes.txt
```

可用 `--target-classes` 只保留关键类别:

```bash
--target-classes person,car,truck,bus
```

当设置目标类别后,`yolo_predictions.json` 会记录过滤后的检测结果;每条检测包含 bbox、类别和置信度,可直接用于后续统计或可视化。

## 17. YOLO 模型来源、输入输出、后处理
默认模型为 Ultralytics YOLOv8n 导出的 ONNX:

```text
models/yolov8n.pt   -> Ultralytics 预训练权重
models/yolov8n.onnx -> scripts/download_yolo_model.py 导出产物
```

推理阶段只使用 ONNX Runtime:

```python
onnxruntime.InferenceSession("models/yolov8n.onnx", providers=["CPUExecutionProvider"])
```

ONNX 张量约定:

```text
input:
  images: [1, 3, 640, 640], tensor(float)

output:
  output0: [1, 84, 8400], tensor(float)
```

前处理:

1. letterbox resize 到 `640x640`,保留宽高比。
2. BGR 转 RGB。
3. 归一化到 `[0, 1]`。
4. HWC 转 CHW,增加 batch 维,输出 `float32 [1, 3, 640, 640]`。

后处理:

1. 解析 YOLOv8 输出 `[1, 84, 8400]`。
2. 按类别分数取 `class_id / confidence`。
3. 使用 `conf_threshold` 过滤低置信度候选。
4. `xywh` 转 `xyxy`。
5. 反 letterbox 映射回原图坐标。
6. clamp bbox 到原图范围。
7. 按类别执行自写 NMS。

实现文件:

```text
app/yolo/preprocess.py
app/yolo/onnx_runner.py
app/yolo/postprocess.py
app/yolo/nms.py
app/yolo/visualizer.py
```

## 18. 检测样本与坏样本导出
报告阶段会导出两类样本索引:

```text
outputs/bad_samples/index.json
outputs/detection_samples/index.json
```

坏样本:

- 来源: `quality_tags` 包含 `bad_quality` 的帧。
- 图像: 原始解码帧。
- 索引字段: `topic / frame_seq / image_path / quality_score / quality_tags / penalties`。
- 上限: `--max-bad-samples`,默认 20。

检测样本:

- 来源: 有 YOLO 检测结果的帧。
- 图像: 绘制 bbox、类别和置信度后的可视化图。
- 索引字段: `topic / frame_seq / image_path / detections`。
- 上限: `--max-detection-samples`,默认 20。

样本文件名会按 Topic 和帧序号生成,便于从报告回溯到原始帧。

## 19. 可观测性 / 日志字段
管线会输出 `[metrics]` 结构化日志行,覆盖关键字段:

```text
mcap_file, topic, message_type, frame_seq, decode_ms, preprocess_ms,
inference_ms, postprocess_ms, quality_score, quality_tags, object_count,
target_object_count, processed_frames, decode_failed_frames,
skipped_low_quality_frames, bad_quality_frames
```

`metrics.json` 会记录:

```text
avg_decode_ms / p95_decode_ms
avg_preprocess_ms / p95_preprocess_ms
avg_inference_ms / p95_inference_ms
avg_postprocess_ms / p95_postprocess_ms
processed_frames_per_sec
```

可用 `--max-frames` 限制处理帧数,用于快速 smoke test 和性能摸底。

## 20. 性能瓶颈说明
CPU 模式下主要瓶颈通常是 YOLO ONNX 推理,其次是图像解码和 letterbox 前处理。
当前实现逐帧迭代 MCAP,不会一次性把全部帧读入内存;报告生成阶段只保留必要的帧级元数据和导出样本。
如果需要提升吞吐,优先考虑:

- 增大 `--sample-every-n`,减少推理帧数
- 保持 `--infer-low-quality false`,让质量门控跳过坏帧
- 使用更小输入尺寸或 OpenVINO 等 CPU 优化后端
- 在 P9/P11 后续阶段增加批处理或多进程 worker

## 21. 异常处理说明
当前异常处理策略:

- MCAP 目录批处理时,单个文件读取失败会写入 `failed_files`,其他文件继续处理。
- 单帧解码失败、未知 encoding、损坏 payload 不会中断管线,会计入 `decoder_stats`。
- 输出目录不存在时自动创建。
- API 参数错误、文件不存在、模型加载失败等会返回 HTTP 400,错误信息放在 `detail.error`。
- YOLO 推理前检查模型和 labels 路径由 ONNX Runtime / 文件读取错误显式暴露。
- 可用 `--max-frames` 快速限制处理量,降低 smoke test 成本。

当前 `app/core/errors.py` 仍保留后续扩展统一异常类型的占位,实际接口已通过 FastAPI `HTTPException` 和管线内部统计处理主要失败路径。

## 22. 已知限制
当前实现限制:

- YOLO 推理为 CPU 单帧串行执行,吞吐主要受 ONNX Runtime 推理耗时影响。
- FastAPI 的质量扫描和 YOLO 推理接口当前同步执行任务,`JobManager` 记录状态,但未接入后台队列或持久化任务存储。
- 配置项主要通过 CLI/API 参数传入,尚未完整接入 pydantic Settings。
- 统一异常类型还未完全抽象,部分错误仍直接由底层库异常转换为 HTTP 400。
- MCAP 读取基于 `rosbags`,当前面向 ROS2 CDR 图像消息。
- 质量规则是启发式规则,适合工程门控和问题解释,不是无参考图像质量模型。
- ONNX 模型默认使用 YOLOv8n COCO 类别,检测精度不作为本作业核心评分目标。

## 23. 复现说明 / AI 使用说明
推荐复现流程:

```bash
cd mcap_yolo_quality_gateway
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_test_mcap.py

python scripts/run_mcap_yolo_inference.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --model ./models/yolov8n.onnx \
  --labels ./models/coco_classes.txt \
  --target-classes person,car,truck,bus \
  --sample-every-n 5 \
  --quality-threshold 0.6 \
  --conf-threshold 0.25 \
  --nms-threshold 0.45 \
  --output-dir ./outputs
```

验收输出:

```text
outputs/mcap_summary.json
outputs/quality_report.json
outputs/quality_report.md
outputs/quality_report.html
outputs/yolo_predictions.json
outputs/metrics.json
outputs/bad_samples/index.json
outputs/detection_samples/index.json
```

Docker 复现:

```bash
docker compose up --build
# 访问 http://127.0.0.1:8000/docs
```

### 提交说明

实际开发耗时: 约 12~18 小时

是否使用 AI 工具: 是

AI 工具使用范围:
用于辅助生成工程骨架、代码实现思路、测试样例、README 文档整理、Docker/脚本验证与问题排查。核心代码、测试结果、Docker 运行和报告输出均以本仓库实际可运行结果为准。

当前已知问题:

1. 仓库内 `sample.mcap` 为合成测试数据,不包含真实 person/car/truck/bus 场景,因此真实 YOLO detections 可能为空。
2. 如果不提交 `models/yolov8n.onnx` 或 `test_data/sample.mcap`,需要按 README 说明放置到 `models/` 和 `test_data/` 后再运行。
3. 宿主机直接运行 `pytest` 需要先安装 `requirements.txt`;Docker 环境中已验证通过。

未完成项:
暂无影响验收的未完成项。

---

## 推荐启动目标(验收)
```bash
docker compose up --build           # 一键启动
# 访问 http://127.0.0.1:8000/docs

python scripts/run_mcap_yolo_inference.py \
  --mcap ./test_data/sample.mcap \
  --auto-detect-topics true \
  --model ./models/yolov8n.onnx \
  --labels ./models/coco_classes.txt \
  --target-classes person,car,truck,bus \
  --sample-every-n 5 \
  --quality-threshold 0.6 \
  --conf-threshold 0.25 \
  --nms-threshold 0.45 \
  --output-dir ./outputs
```

## 参考资料
- MCAP ROS2: https://mcap.dev/guides/getting-started/ros-2
- rosbag2_storage_mcap: https://docs.ros.org/en/humble/p/rosbag2_storage_mcap/
- ONNX Runtime: https://onnxruntime.ai/docs/
- Ultralytics YOLO export: https://docs.ultralytics.com/modes/export/
