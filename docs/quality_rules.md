# 质量评分规则

本项目对每一帧解码后的 BGR 图像计算可解释的质量指标,再将问题映射为扣分项。
默认阈值用于测试数据和 CPU demo,后续可在 CLI/API 层暴露为配置。

## 单帧指标

| 指标 | 定义 | 用途 |
| --- | --- | --- |
| `width` / `height` | 图像宽高 | 判断低分辨率 |
| `aspect_ratio` | `width / height` | 报告中保留,便于发现异常画幅 |
| `brightness_mean` | 灰度图像素均值,范围 0~255 | 判断过暗/过亮 |
| `brightness_std` | 灰度图标准差 | 作为对比度基础 |
| `contrast_score` | 当前等同于 `brightness_std` | 判断低对比度 |
| `blur_score` | 灰度图 Laplacian 方差 | 判断模糊 |
| `saturation_mean` | HSV S 通道均值,范围 0~255 | 报告中保留,便于发现颜色异常 |
| `is_empty` | 标准差极低且接近全黑/全白 | 判断空帧 |
| `is_corrupted` | 空帧、非法 ndarray 或无法计算指标 | 判断损坏帧 |

## 默认问题阈值

| 标志位 | 默认规则 |
| --- | --- |
| `is_too_dark` | `brightness_mean <= 35` |
| `is_too_bright` | `brightness_mean >= 245` |
| `is_blurry` | `blur_score < 80`,但空帧不重复标模糊 |
| `is_low_contrast` | `contrast_score < 18` |
| `is_low_resolution` | `width < 320` 或 `height < 240` |
| `is_corrupted` | `is_empty == true` 或指标计算失败 |

## 评分公式

基础分为 1.0:

```text
score = 1.0
  - blur_penalty
  - exposure_penalty
  - contrast_penalty
  - resolution_penalty
  - corruption_penalty
  - timestamp_penalty
```

最终 `score` 会裁剪到 `[0, 1]`。

| 扣分项 | 默认权重 |
| --- | --- |
| `blur_penalty` | 模糊帧扣 `0.20` |
| `exposure_penalty` | 过暗或过亮扣 `0.20` |
| `contrast_penalty` | 低对比度扣 `0.15` |
| `resolution_penalty` | 低分辨率扣 `0.25`,轻微低于阈值可扣 `0.10` |
| `corruption_penalty` | 损坏/空帧扣 `0.45` |
| `timestamp_penalty` | 时序模块传入,默认 `0.0`,最大 `0.20` |

## 标签规则

每个问题会生成对应标签:

```text
corrupted, too_dark, too_bright, blurry, low_contrast, low_resolution
```

当 `score < quality_threshold` 时追加:

```text
bad_quality
```

默认 `quality_threshold = 0.6`。报告会同时输出 `score`、`quality_tags` 和各项 `penalties`,保证每个低质量判定都能追溯到具体原因。
