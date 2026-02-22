# services/achievement.py
"""周刊成就检测服务"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path
from enum import Enum

import pandas as pd

from common.logger import logger
from common.io import save_excel
from common.config import get_app_config, get_paths


class AchiType(Enum):
    """成就类型"""

    EMERGING_HIT = "Emerging Hit!"
    MEGA_HIT = "Mega Hit!!!"
    POTENTIAL_REGULAR = "门番候补"
    REGULAR = "门番"


@dataclass
class AchiDef:
    """成就定义"""

    name: str
    weeks: int
    rank: int
    threshold: Optional[int] = None  # 仅 REGULAR 类型需要


@dataclass
class AchievedSong:
    """达成成就的歌曲"""

    name: str
    period: int
    honor: str
    title: str = ""
    bvid: str = ""
    author: str = ""
    pubdate: str = ""
    progress: str = ""


# ==================== 成就定义 ====================

ACHIEVEMENT_DEFINITIONS: Dict[AchiType, AchiDef] = {
    AchiType.EMERGING_HIT: AchiDef(
        name="Emerging Hit!",
        weeks=3,
        rank=5,
    ),
    AchiType.MEGA_HIT: AchiDef(
        name="Mega Hit!!!",
        weeks=5,
        rank=3,
    ),
    AchiType.POTENTIAL_REGULAR: AchiDef(
        name="门番候补",
        weeks=15,
        rank=20,
        threshold=10,
    ),
    AchiType.REGULAR: AchiDef(
        name="门番",
        weeks=30,
        rank=20,
        threshold=20,
    ),
}


def get_window_size() -> int:
    """获取需要的历史窗口大小"""
    return max(d.weeks for d in ACHIEVEMENT_DEFINITIONS.values())


# ==================== 成就检测器 ====================


class AchievementDetector:
    """成就检测器"""

    def __init__(self, definitions: Dict[AchiType, AchiDef] = None):
        self.definitions = definitions or ACHIEVEMENT_DEFINITIONS

    def detect(self, history: deque, name: str) -> Set[AchiType]:
        """检测某首歌曲达成的所有成就"""
        achieved: Set[AchiType] = set()
        hist_list = list(history)

        for achi_type, definition in self.definitions.items():
            weeks = definition.weeks
            rank = definition.rank
            hist_slice = hist_list[-weeks:]

            if achi_type in {AchiType.EMERGING_HIT, AchiType.MEGA_HIT}:
                # 连续在榜
                if len(hist_slice) == weeks and all(
                    name in week[:rank] for week in hist_slice if week
                ):
                    achieved.add(achi_type)
            else:
                # 累计在榜
                count = sum(name in week[:rank] for week in hist_slice if week)
                if definition.threshold and count >= definition.threshold:
                    achieved.add(achi_type)

        return achieved

    def calculate_progress(self, achi_type: AchiType, history: deque, name: str) -> str:
        """计算成就进度"""
        definition = self.definitions[achi_type]
        hist_slice = list(history)[-definition.weeks :]
        rank = definition.rank

        if achi_type in {AchiType.EMERGING_HIT, AchiType.MEGA_HIT}:
            # 显示每周排名
            ranks = []
            for week in hist_slice:
                if name in week[:rank]:
                    ranks.append(str(week.index(name) + 1))
                elif name in week:
                    ranks.append("X")
                else:
                    ranks.append("-")
            return "~".join(ranks)
        else:
            # 显示累计次数
            count = sum(name in week[:rank] for week in hist_slice if week)
            first_idx = next(
                (
                    i
                    for i, week in enumerate(hist_slice)
                    if week and name in week[:rank]
                ),
                -1,
            )
            elapsed = (
                len(hist_slice) - first_idx if first_idx != -1 else len(hist_slice)
            )
            return f"{count}/{elapsed}"


# ==================== 成就管理器 ====================


@dataclass
class PeriodData:
    """期刊数据"""

    names: List[str]
    details: pd.DataFrame
    date_str: str


class AchievementManager:
    """成就管理器"""

    REQUIRED_COLS = ["name", "title", "bvid", "author", "pubdate"]
    OUTPUT_COLS = ["title", "bvid", "name", "author", "pubdate", "honor"]

    def __init__(
        self,
        output_dir: Path,
        window_size: int = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.window_size = window_size or get_window_size()
        self.detector = AchievementDetector()
        self.history: deque = deque(maxlen=self.window_size)

        self.master_file = self.output_dir / "成就.xlsx"
        self.master_db = self._load_master_db()

    def _load_master_db(self) -> Dict[AchiType, Dict[str, Tuple[int, str]]]:
        """加载主数据库"""
        db: Dict[AchiType, Dict[str, Tuple[int, str]]] = defaultdict(dict)

        if not self.master_file.exists():
            return db

        try:
            with pd.ExcelFile(self.master_file, engine="openpyxl") as xls:
                for achi_type in AchiType:
                    sheet = achi_type.value
                    if sheet in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet)
                        for _, row in df.iterrows():
                            db[achi_type][str(row["name"])] = (
                                int(row["index"]),
                                str(row.get("progress", "")),
                            )
        except Exception as e:
            logger.warning(f"加载成就数据库失败: {e}")

        return db

    def _save_master_db(self):
        """保存主数据库"""
        with pd.ExcelWriter(self.master_file, engine="openpyxl") as writer:
            for achi_type, song_map in self.master_db.items():
                if song_map:
                    rows = [
                        {"name": name, "index": idx, "progress": progress}
                        for name, (idx, progress) in sorted(
                            song_map.items(), key=lambda x: (x[1][0], x[0])
                        )
                    ]
                    pd.DataFrame(rows).to_excel(
                        writer, sheet_name=achi_type.value, index=False
                    )

    def _save_report(self, date_str: str, records: List[AchievedSong]):
        """保存当期成就报告"""
        output_file = self.output_dir / f"成就{date_str}.xlsx"

        if not records:
            logger.info(f"当期 {date_str} 没有新成就")
            df = pd.DataFrame(columns=self.OUTPUT_COLS)
        else:
            rows = [
                {
                    "title": r.title,
                    "bvid": r.bvid,
                    "name": r.name,
                    "author": r.author,
                    "pubdate": r.pubdate,
                    "honor": r.honor,
                }
                for r in sorted(records, key=lambda x: (x.honor, x.name))
            ]
            df = pd.DataFrame(rows, columns=self.OUTPUT_COLS)

        save_excel(df, output_file)
        logger.info(f"已保存成就报告: {output_file.name}")

    def _get_song_details(
        self, name: str, details: pd.DataFrame
    ) -> Tuple[str, str, str, str]:
        """获取歌曲详情"""
        if details.empty:
            return "", "", "", ""

        row = details[details["name"] == name]
        if row.empty:
            return "", "", "", ""

        data = row.iloc[0]
        title = str(data.get("title", ""))
        bvid = str(data.get("bvid", ""))
        author = str(data.get("author", ""))
        pubdate = data.get("pubdate")

        if pd.notna(pubdate):
            try:
                pubdate = pd.to_datetime(pubdate).strftime("%Y-%m-%d %H:%M:%S")
            except:
                pubdate = str(pubdate)
        else:
            pubdate = ""

        return title, bvid, author, pubdate

    def process_period(
        self, period: int, data: PeriodData, is_target: bool = False
    ) -> List[AchievedSong]:
        """处理单期数据"""
        self.history.append(data.names)

        if not is_target:
            return []

        records: List[AchievedSong] = []
        unique_songs = {name for week in self.history for name in week if name}

        for name in unique_songs:
            achieved_types = self.detector.detect(self.history, name)

            for achi_type in achieved_types:
                if name in self.master_db[achi_type]:
                    continue  # 已达成过

                progress = self.detector.calculate_progress(
                    achi_type, self.history, name
                )
                title, bvid, author, pubdate = self._get_song_details(
                    name, data.details
                )

                record = AchievedSong(
                    name=name,
                    period=period,
                    honor=achi_type.value,
                    title=title,
                    bvid=bvid,
                    author=author,
                    pubdate=pubdate,
                    progress=progress,
                )
                records.append(record)
                self.master_db[achi_type][name] = (period, progress)
                logger.info(f"新成就: '{name}' 期{period} {achi_type.value} {progress}")

        self._save_report(data.date_str, records)
        return records

    def run(
        self,
        period_data: Dict[int, PeriodData],
        start_period: int,
        target_period: int,
    ):
        """执行成就检测"""
        all_records: List[AchievedSong] = []

        for period in range(start_period, target_period + 1):
            data = period_data[period]
            is_target = period == target_period
            records = self.process_period(period, data, is_target)
            all_records.extend(records)

        self._save_master_db()
        return all_records


# ==================== 工具函数 ====================


def load_ranking_file(file_path: Path) -> PeriodData:
    """加载榜单文件"""
    df = pd.read_excel(file_path, engine="openpyxl")
    top20 = df.head(20).copy()

    names = []
    for idx in top20.index:
        name = top20.loc[idx, "name"]
        name = str(name).strip() if pd.notna(name) else ""
        names.append(name)
        top20.loc[idx, "name"] = name

    required = ["name", "title", "bvid", "author", "pubdate"]
    details = top20[[c for c in required if c in top20.columns]]

    # 从文件名提取日期
    stem = file_path.stem
    date_str = stem.split(" ")[-1] if " " in stem else stem

    return PeriodData(names=names, details=details, date_str=date_str)


def scan_ranking_files(input_dir: Path) -> List[Tuple[Path, datetime]]:
    """扫描并排序榜单文件"""
    files = []
    for file_path in input_dir.glob("*.xlsx"):
        stem = file_path.stem
        date_str = stem.split(" ")[-1] if " " in stem else stem
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            files.append((file_path, date))
        except ValueError:
            continue

    files.sort(key=lambda x: x[1])
    return files


def run_achievement_detection(
    input_dir: Path = None,
    output_dir: Path = None,
    start_file: str = None,
    start_index: int = None,
):
    """
    执行成就检测

    Args:
        input_dir: 榜单文件目录（可选，优先级高于配置）
        output_dir: 输出目录（可选，优先级高于配置）
        start_file: 起始文件（可选，优先级高于配置）
        start_index: 起始期数（可选，优先级高于配置）
        config: 成就配置（可选，自动加载）
    """
    # 加载配置
    config = get_app_config()
    paths = get_paths()

    # 参数优先级：显式参数 > 配置文件
    input_dir = input_dir or paths.weekly_main
    output_dir = output_dir or paths.achievement
    start_file = start_file or config.achievement.start_file
    start_index = (
        start_index if start_index is not None else config.achievement.start_index
    )

    window_size = get_window_size()
    all_files = scan_ranking_files(input_dir)

    if not all_files:
        logger.error(f"未找到榜单文件: {input_dir}")
        return

    # 定位起始文件
    start_date_str = Path(start_file).stem.split(" ")[-1]
    file_start_idx = next(
        (
            i
            for i, (_, dt) in enumerate(all_files)
            if dt.strftime("%Y-%m-%d") == start_date_str
        ),
        0,
    )

    # 计算期数范围
    latest_period = start_index + (len(all_files) - 1 - file_start_idx)
    target_period = latest_period
    range_start = max(start_index, target_period - window_size + 1)

    logger.info(f"目标期数: {target_period}, 窗口: {range_start}-{target_period}")

    # 加载数据
    period_data: Dict[int, PeriodData] = {}
    for period in range(range_start, target_period + 1):
        file_idx = file_start_idx + (period - start_index)
        file_path, _ = all_files[file_idx]
        period_data[period] = load_ranking_file(file_path)

    # 执行检测
    manager = AchievementManager(output_dir=output_dir, window_size=window_size)
    manager.run(period_data, range_start, target_period)
