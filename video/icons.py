# video/icons.py
"""SVG 图标渲染器"""

import tempfile
from pathlib import Path
from typing import Dict, Optional

import yaml
import cairosvg


class IconRenderer:
    """SVG 图标渲染器"""

    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config/icons.yaml")
        self._icons: Optional[Dict] = None
        self._render_config: Optional[Dict] = None
        self._cache: Dict[str, Path] = {}
        self._temp_dir: Optional[Path] = None

    def _load_config(self):
        if self._icons is None:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._icons = data.get("icons", {})
            self._render_config = data.get("render", {})

    def _ensure_temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="icons_"))
        return self._temp_dir

    def render(self, name: str, size: int = None, color: str = None) -> Path:
        """渲染图标为 PNG 并返回路径"""
        self._load_config()

        size = size or self._render_config.get("size", 30)
        color = color or self._render_config.get("color", "#DCDCDC")

        cache_key = f"{name}_{size}_{color}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        icon_data = self._icons.get(name)
        if not icon_data:
            raise ValueError(f"未知图标: {name}")

        viewBox = icon_data.get("viewBox", "0 0 24 24")
        path_d = icon_data.get("path", "")
        fill_rule = icon_data.get("fillRule", "")
        clip_rule = icon_data.get("clipRule", "")
        path_attrs = f'd="{path_d}" fill="{color}"'
        if fill_rule:
            path_attrs += f' fill-rule="{fill_rule}"'
        if clip_rule:
            path_attrs += f' clip-rule="{clip_rule}"'

        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="{viewBox}" width="{size}" height="{size}">
            <path {path_attrs}/>
        </svg>"""

        temp_dir = self._ensure_temp_dir()
        output_path = temp_dir / f"{name}_{size}.png"

        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(output_path))

        self._cache[cache_key] = output_path
        return output_path

    def get_all(self, size: int = None) -> Dict[str, Path]:
        """获取所有图标路径"""
        self._load_config()
        return {name: self.render(name, size) for name in self._icons.keys()}


# 全局单例
_renderer: Optional[IconRenderer] = None


def get_icon_renderer() -> IconRenderer:
    global _renderer
    if _renderer is None:
        _renderer = IconRenderer()
    return _renderer
