# MCAP 输入说明

MCAP 读取基于 `rosbags`,支持单文件和目录批处理。

常用参数:

- `--mcap`: 指定单个 MCAP 文件。
- `--mcap-dir`: 递归扫描目录中的 `.mcap` 文件。
- `--auto-detect-topics`: 自动识别图像 Topic。
- `--topics`: 手动指定图像 Topic。
- `--sample-every-n`: 每 N 帧采样 1 帧。
- `--start-sec` / `--end-sec`: 按相对起始时间裁剪处理区间。

测试数据可通过以下命令生成:

```bash
python scripts/generate_test_mcap.py
```
