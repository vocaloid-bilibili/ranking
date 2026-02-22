# video/config.py
"""视频模块独立配置管理"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Any

# 视频模块根目录
VIDEO_MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VIDEO_MODULE_ROOT.parent
VIDEO_CONFIG_PATH = PROJECT_ROOT / "config" / "video.yaml"


def _load_yaml(path: Path = VIDEO_CONFIG_PATH) -> Dict[str, Any]:
    """加载 YAML 配置"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class VideoPathsConfig:
    """视频路径配置"""

    # 数据源
    daily_ranking_main: Path
    daily_ranking_new: Path
    milestone: Path
    milestone_100k: Path

    # 视频专用
    videos_cache: Path
    daily_video_output: Path

    @classmethod
    def from_yaml(cls, config: Dict[str, Any] = None) -> "VideoPathsConfig":
        if config is None:
            config = _load_yaml()

        p = config.get("paths", {})
        milestone = PROJECT_ROOT / p.get("milestone", "data/features/milestone")

        return cls(
            daily_ranking_main=PROJECT_ROOT
            / p.get("daily_ranking_main", "data/daily/ranking/main"),
            daily_ranking_new=PROJECT_ROOT
            / p.get("daily_ranking_new", "data/daily/ranking/new"),
            milestone=milestone,
            milestone_100k=milestone / "100k",
            videos_cache=PROJECT_ROOT / p.get("videos_cache", "downloads/videos"),
            daily_video_output=PROJECT_ROOT
            / p.get("daily_video_output", "export/daily_video"),
        )

    def ensure_dirs(self):
        self.videos_cache.mkdir(parents=True, exist_ok=True)
        self.daily_video_output.mkdir(parents=True, exist_ok=True)
        self.milestone_100k.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FontsConfig:
    """字体配置"""

    regular: str
    bold: str

    @classmethod
    def from_yaml(cls, config: Dict[str, Any] = None) -> "FontsConfig":
        if config is None:
            config = _load_yaml()

        f = config.get("fonts", {})
        return cls(
            regular=str(f.get("regular", "C:/Windows/Fonts/msyh.ttc")),
            bold=str(f.get("bold", "C:/Windows/Fonts/msyhbd.ttc")),
        )


@dataclass(frozen=True)
class FfmpegConfig:
    """FFmpeg 配置"""

    bin: str

    @classmethod
    def from_yaml(cls, config: Dict[str, Any] = None) -> "FfmpegConfig":
        if config is None:
            config = _load_yaml()

        return cls(bin=str(config.get("ffmpeg", {}).get("bin", "ffmpeg")))


@dataclass(frozen=True)
class VideoParamsConfig:
    """视频参数配置"""

    top_n: int
    clip_duration: float
    first_issue_date: str
    fps: int

    @classmethod
    def from_yaml(cls, config: Dict[str, Any] = None) -> "VideoParamsConfig":
        if config is None:
            config = _load_yaml()

        v = config.get("video", {})
        return cls(
            top_n=int(v.get("top_n", 10)),
            clip_duration=float(v.get("clip_duration", 15.0)),
            first_issue_date=str(v.get("first_issue_date", "20240907")),
            fps=int(v.get("fps", 60)),
        )


@dataclass(frozen=True)
class UiConfig:
    """UI 配置"""

    scroll_bg_color: Tuple[int, int, int, int]
    card_width: int
    card_height: int
    card_gap: int
    card_radius: int
    scroll_hold_time: float
    scroll_speed_pps: float

    @classmethod
    def from_yaml(cls, config: Dict[str, Any] = None) -> "UiConfig":
        if config is None:
            config = _load_yaml()

        u = config.get("ui", {})
        c = u.get("scroll_bg_color", [0, 139, 139, 255])
        bg_color = tuple(c) if len(c) == 4 else (c[0], c[1], c[2], 255)

        return cls(
            scroll_bg_color=bg_color,
            card_width=int(u.get("card_width", 960)),
            card_height=int(u.get("card_height", 200)),
            card_gap=int(u.get("card_gap", 20)),
            card_radius=int(u.get("card_radius", 20)),
            scroll_hold_time=float(u.get("scroll_hold_time", 3.0)),
            scroll_speed_pps=float(u.get("scroll_speed_pps", 150.0)),
        )


@dataclass(frozen=True)
class VideoConfig:
    """视频模块完整配置"""

    project_root: Path
    paths: VideoPathsConfig
    fonts: FontsConfig
    ffmpeg: FfmpegConfig
    video: VideoParamsConfig
    ui: UiConfig
    weekday_colors: Dict[int, str]

    @classmethod
    def load(cls, config_path: Path = VIDEO_CONFIG_PATH) -> "VideoConfig":
        """加载完整配置"""
        config = _load_yaml(config_path)

        # 星期颜色
        weekday_colors = {}
        for k, v in config.get("weekday_colors", {}).items():
            weekday_colors[int(k)] = str(v)

        # 默认颜色
        default_colors = {
            0: "#8C4E70",
            1: "#D66547",
            2: "#595959",
            3: "#4992A7",
            4: "#BDBDBD",
            5: "#C48700",
            6: "#55CCCC",
        }
        for k, v in default_colors.items():
            if k not in weekday_colors:
                weekday_colors[k] = v

        return cls(
            project_root=PROJECT_ROOT,
            paths=VideoPathsConfig.from_yaml(config),
            fonts=FontsConfig.from_yaml(config),
            ffmpeg=FfmpegConfig.from_yaml(config),
            video=VideoParamsConfig.from_yaml(config),
            ui=UiConfig.from_yaml(config),
            weekday_colors=weekday_colors,
        )


# 便捷加载函数
def load_video_config() -> VideoConfig:
    """加载视频配置"""
    return VideoConfig.load()
