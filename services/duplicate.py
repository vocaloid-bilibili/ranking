# services/duplicate.py
"""疑似重复曲目检测服务"""

import hashlib
import io
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
import imagehash
from PIL import Image
from tqdm import tqdm

from common.io import save_excel
from common.config import get_app_config, get_paths
from common.logger import logger


@dataclass
class DuplicateConfig:
    """重复检测配置"""

    # 感知哈希相似度阈值
    hash_threshold: int = 15
    # 日期差距阈值 (天)
    date_threshold_days: int = 3
    # 图片缓存目录
    image_cache_dir: Path = Path("downloads/image_url")
    # 输入文件
    input_file: Path = Path("data/collected.xlsx")
    # 输出文件
    output_file: Path = Path("export/duplicate.xlsx")


@dataclass
class DuplicateResult:
    """检测结果"""

    name_duplicates: Set[int] = field(default_factory=set)
    image_similar: Set[int] = field(default_factory=set)
    date_close: Set[int] = field(default_factory=set)

    @property
    def all_indices(self) -> Set[int]:
        return self.name_duplicates | self.image_similar | self.date_close

    def get_reasons(self, idx: int) -> List[str]:
        reasons = []
        if idx in self.name_duplicates:
            reasons.append("相同曲名")
        if idx in self.image_similar:
            reasons.append("图片相似")
        if idx in self.date_close:
            reasons.append("日期相近")
        return reasons


class ImageHashCache:
    """图片哈希缓存"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hash_cache: Dict[str, Optional[imagehash.ImageHash]] = {}

    def get_hash(self, url: str) -> Optional[imagehash.ImageHash]:
        """获取图片的感知哈希"""
        if not isinstance(url, str) or not url.startswith("http"):
            return None

        if url in self._hash_cache:
            return self._hash_cache[url]

        url_hash = hashlib.sha256(url.encode()).hexdigest()
        cache_path = self.cache_dir / f"{url_hash}.jpg"

        try:
            if cache_path.exists():
                with Image.open(cache_path) as img:
                    h = imagehash.phash(img)
            else:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                cache_path.write_bytes(response.content)
                with Image.open(io.BytesIO(response.content)) as img:
                    h = imagehash.phash(img)

            self._hash_cache[url] = h
            return h
        except Exception:
            self._hash_cache[url] = None
            return None


class DuplicateDetector:
    """重复曲目检测器"""

    def __init__(self, config: DuplicateConfig = None):
        self.config = config or DuplicateConfig()
        self._image_cache: Optional[ImageHashCache] = None

    @property
    def image_cache(self) -> ImageHashCache:
        if self._image_cache is None:
            self._image_cache = ImageHashCache(self.config.image_cache_dir)
        return self._image_cache

    def detect(self, df: pd.DataFrame) -> DuplicateResult:
        """执行完整检测"""
        result = DuplicateResult()

        # 预处理
        df = self._preprocess(df)

        # 三种检测方式
        result.name_duplicates = self._find_name_duplicates(df)
        result.image_similar = self._find_similar_images(df)
        result.date_close = self._find_close_pubdates(df)

        return result

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        df = df.copy()
        for col in ["name", "author", "image_url", "uploader"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("")
        return df

    def _normalize_multi_value(self, value: str, separator: str = "、") -> str:
        """标准化多值字段"""
        if not isinstance(value, str):
            return str(value)
        parts = [p.strip() for p in value.split(separator) if p.strip()]
        return ",".join(sorted(parts))

    def _find_name_duplicates(self, df: pd.DataFrame) -> Set[int]:
        """查找相同曲名的重复"""
        logger.info("查找相同曲名...")

        key_cols = ["author", "synthesizer", "vocal", "type", "copyright"]
        multi_value_cols = ["author", "synthesizer", "vocal"]
        other_key_cols = ["author", "synthesizer", "vocal", "type"]

        duplicates = df[df.duplicated(subset=["name"], keep=False)]
        if duplicates.empty:
            return set()

        found = set()

        for _, group in duplicates.groupby("name"):
            compare = group[key_cols].copy()

            # 标准化多值字段
            for col in multi_value_cols:
                compare[col] = (
                    compare[col].astype(str).apply(self._normalize_multi_value)
                )

            compare["type"] = compare["type"].astype(str).str.strip()
            compare["copyright"] = (
                pd.to_numeric(compare["copyright"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

            # 检查版权争议
            copyright_values = set(compare["copyright"].unique())
            is_copyright_controversial = 1 in copyright_values and bool(
                copyright_values.intersection({2, 3, 4})
            )

            # 检查其他字段差异
            other_tuples = [
                tuple(rec) for rec in compare[other_key_cols].to_records(index=False)
            ]
            is_other_controversial = len(set(other_tuples)) > 1

            if is_copyright_controversial or is_other_controversial:
                found.update(group.index.tolist())

        logger.info(f"曲名重复: {len(found)} 条")
        return found

    def _find_similar_images(self, df: pd.DataFrame) -> Set[int]:
        """通过图片相似度查找重复"""
        logger.info("查找图片相似...")

        found = set()
        grouped = df.groupby("author")

        for _, group in tqdm(grouped, desc="分析图片"):
            if len(group) < 2:
                continue

            for i, j in combinations(group.index, 2):
                row1, row2 = group.loc[i], group.loc[j]

                # 跳过同名或同UP
                if row1["name"] == row2["name"] or row1["uploader"] == row2["uploader"]:
                    continue

                hash1 = self.image_cache.get_hash(row1["image_url"])
                hash2 = self.image_cache.get_hash(row2["image_url"])

                if hash1 and hash2:
                    distance = hash1 - hash2
                    if distance <= self.config.hash_threshold:
                        found.add(i)
                        found.add(j)

        logger.info(f"图片相似: {len(found)} 条")
        return found

    def _find_close_pubdates(self, df: pd.DataFrame) -> Set[int]:
        """通过发布日期相近查找重复"""
        logger.info("查找日期相近...")

        found = set()
        df = df.copy()
        df["_pubdate_dt"] = pd.to_datetime(df["pubdate"], errors="coerce")

        grouped = df.groupby("author")
        threshold = pd.Timedelta(days=self.config.date_threshold_days)

        for _, group in tqdm(grouped, desc="分析日期"):
            if len(group) < 2:
                continue

            for i, j in combinations(group.index, 2):
                row1, row2 = group.loc[i], group.loc[j]

                # 跳过同名或同UP或日期缺失
                if (
                    row1["name"] == row2["name"]
                    or row1["uploader"] == row2["uploader"]
                    or pd.isna(row1["_pubdate_dt"])
                    or pd.isna(row2["_pubdate_dt"])
                ):
                    continue

                time_diff = abs(row1["_pubdate_dt"] - row2["_pubdate_dt"])
                if time_diff <= threshold:
                    found.add(i)
                    found.add(j)

        logger.info(f"日期相近: {len(found)} 条")
        return found


def run_duplicate_detection(
    input_file: Path = None,
    output_file: Path = None,
):
    app_config = get_app_config()
    paths = get_paths()

    input_file = input_file or paths.collected
    output_file = output_file or (paths.export / "duplicate.xlsx")

    config = DuplicateConfig(
        hash_threshold=app_config.duplicate.hash_threshold,
        date_threshold_days=app_config.duplicate.date_threshold_days,
        image_cache_dir=paths.downloads_image,
        input_file=input_file,
        output_file=output_file,
    )

    # 检查输入
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return

    # 读取数据
    try:
        df = pd.read_excel(input_file, dtype={"aid": str})
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return

    # 执行检测
    detector = DuplicateDetector(config)
    result = detector.detect(df)

    # 整理结果
    all_indices = result.all_indices
    if not all_indices:
        logger.info("未找到任何疑似重复记录")
        return

    final_df = df.loc[list(all_indices)].copy()
    final_df["reason"] = final_df.index.map(
        lambda idx: "；".join(sorted(result.get_reasons(idx)))
    )

    # 排序
    final_df = final_df.sort_values(by=["author", "name", "pubdate"])

    # 调整列顺序
    cols = final_df.columns.tolist()
    if "reason" in cols:
        cols.remove("reason")
        cols.append("reason")
        final_df = final_df[cols]

    logger.info(f"共找到 {len(final_df)} 条疑似重复记录")

    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_excel(final_df, output_file)
    logger.info(f"已保存: {output_file}")
