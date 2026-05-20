# common/config.py
"""统一配置管理"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ==================== 配置加载器 ====================


class ConfigLoader:
    """配置加载器（单例）"""

    _instance: Optional["ConfigLoader"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, path: str = "config/app.yaml"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(path)
        return cls._instance

    def _load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, *keys, default=None):
        result = self._config
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key)
            else:
                return default
            if result is None:
                return default
        return result

    @property
    def raw(self) -> Dict[str, Any]:
        return self._config

    @classmethod
    def reload(cls, path: str = "config/app.yaml"):
        cls._instance = None
        return cls(path)


def get_config() -> ConfigLoader:
    return ConfigLoader()


# ==================== 路径配置 ====================


@dataclass(frozen=True)
class Paths:
    """路径配置"""

    collected: Path
    snapshot_main: Path
    snapshot_new: Path
    special_data: Path
    keywords: Path
    columns: Path
    special_config: Path

    @classmethod
    def load(cls) -> "Paths":
        cfg = get_config()
        paths = cfg.get("paths", default={})
        return cls(**{k: Path(paths[k]) for k in cls.__dataclass_fields__})

    def ensure_dirs(self):
        for name in self.__dataclass_fields__:
            path = getattr(self, name)
            if isinstance(path, Path) and not path.suffix:
                path.mkdir(parents=True, exist_ok=True)

    def load_keywords(self) -> List[str]:
        return json.loads(self.keywords.read_text(encoding="utf-8"))

    def load_usecols(self, key: str) -> List[str]:
        data = json.loads(self.columns.read_text(encoding="utf-8"))
        return data.get("columns", {}).get(key, [])


# ==================== 全局实例 ====================

_paths: Optional[Paths] = None


def get_paths() -> Paths:
    global _paths
    if _paths is None:
        _paths = Paths.load()
    return _paths
