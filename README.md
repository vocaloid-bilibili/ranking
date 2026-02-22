# 术力口数据库

B 站虚拟歌手排行榜数据处理系统，支持日刊/周刊/月刊生成、视频制作、JSON 导出等功能。

## 目录结构

```
├── bilibili/          # B站API客户端
├── common/            # 通用模块
├── ranking/           # 排行榜计算
├── services/          # 功能服务
├── video/             # 视频生成
├── config/            # 配置文件
├── data/              # 数据存储
│   ├── snapshot/      # 每日快照
│   ├── daily/         # 日刊数据
│   ├── weekly/        # 周刊数据
│   ├── monthly/       # 月刊数据
│   └── features/      # 成就/里程碑等
├── downloads/         # 下载缓存
└── export/            # 导出文件
```

## 配置

主配置文件 `config/app.yaml`，包含路径、模板、周期参数等。

其他配置：

- `keywords.json` - 搜索关键词
- `usecols.json` - 列定义
- `ai.yaml` - AI 标注配置
- `video.yaml` - 视频生成配置

## 日常流程

### 1. 数据抓取（每日 0 点）

```bash
python 抓取数据.py       # 已收录曲目
python 抓取新曲数据.py   # 新曲搜索
```

### 2. 差异计算

```bash
python 计算数据.py
```

生成 `data/daily/diff/` 下的差异文件。

### 3. 新曲标注

手动或 AI 辅助：

```bash
python AI打标.py
```

需核对字段：`name`、`author`、`synthesizer`、`vocal`、`type`、`copyright`

### 4. 日刊生成

```bash
python 合并.py           # 合并差异 + 更新收录
python 新曲排行榜.py     # 生成新曲榜
```

## 周刊/月刊

```bash
python 周刊.py
python 月刊.py
```

按提示输入日期，输出到 `data/weekly/` 或 `data/monthly/`。

## 辅助功能

| 脚本              | 功能                                |
| ----------------- | ----------------------------------- |
| `成就检测.py`     | 检测 Emerging Hit / Mega Hit / 门番 |
| `百万达成.py`     | 统计播放量突破百万的视频            |
| `重复检测.py`     | 检查疑似重复收录                    |
| `补筛.py`         | 从热榜补充遗漏视频                  |
| `抓取特殊数据.py` | 特刊数据抓取                        |

## 视频生成

```bash
python 日刊视频版.py
```

自动下载素材、裁剪高潮、叠加信息、生成封面、拼接成片。

配置见 `config/video.yaml`。

## JSON 导出

```bash
python 导出周刊.py
python 导出月刊.py
```

输出到 `export/json/`，供前端使用。

## SFTP 同步

```bash
python 下载.py    # 从服务器拉取数据
python 上传.py    # 推送到服务器
```

配置 `config/sftp.yaml`。

## 依赖

```bash
pip install -r requirements.txt
```

主要依赖：

- `pandas` / `openpyxl` - 数据处理
- `aiohttp` / `bilibili-api-python` - API 请求
- `yt-dlp` / `ffmpeg` - 视频下载处理
- `librosa` / `demucs` - 音频分析
- `Pillow` / `cairosvg` - 图像处理
