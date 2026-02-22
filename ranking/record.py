# ranking/record.py
"""记录处理模块 - 批量计算评分"""

from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from common.logger import logger
from common.models import VideoStats, ScoreResult, RankingType
from ranking.calculator import calculate_score


STAT_FIELDS = ["view", "favorite", "coin", "like", "danmaku", "reply", "share"]
METADATA_FIELDS = [
    "title",
    "bvid",
    "aid",
    "name",
    "author",
    "uploader",
    "copyright",
    "synthesizer",
    "vocal",
    "type",
    "pubdate",
    "duration",
    "page",
    "image_url",
    "intro",
    "tid",
]
COLLECTED_MERGE_FIELDS = ["name", "author", "synthesizer", "copyright", "vocal", "type"]


def _build_output_dict(record: pd.Series, result: ScoreResult) -> Dict[str, Any]:
    """构建输出字典"""
    output = {}

    # 元数据
    for field in METADATA_FIELDS:
        if field in record and pd.notna(record.get(field)):
            output[field] = record[field]

    # 统计增量
    output.update(
        {
            "view": result.view,
            "favorite": result.favorite,
            "coin": result.coin,
            "like": result.like,
            "danmaku": result.danmaku,
            "reply": result.reply,
            "share": result.share,
        }
    )

    # 评分系数（格式化）
    output.update(
        {
            "viewR": f"{result.view_rate:.2f}",
            "favoriteR": f"{result.favorite_rate:.2f}",
            "coinR": f"{result.coin_rate:.2f}",
            "likeR": f"{result.like_rate:.2f}",
            "danmakuR": f"{result.danmaku_rate:.2f}",
            "replyR": f"{result.reply_rate:.2f}",
            "shareR": f"{result.share_rate:.2f}",
        }
    )

    # 修正系数
    output.update(
        {
            "fixA": f"{result.fix_a:.2f}",
            "fixB": f"{result.fix_b:.2f}",
            "fixC": f"{result.fix_c:.2f}",
            "fixD": f"{result.fix_d:.2f}",
        }
    )

    output["point"] = result.point
    return output


class RecordProcessor:
    """记录处理器"""

    def __init__(
        self,
        ranking_type: RankingType,
        use_old_data: bool = False,
        old_time_threshold: Optional[str] = None,
    ):
        self.ranking_type = ranking_type
        self.use_old_data = use_old_data
        self.old_time_threshold = old_time_threshold
        self._old_index: Dict[str, pd.Series] = {}
        self._collected_index: Dict[str, pd.Series] = {}

    def _build_index(self, df: Optional[pd.DataFrame]) -> Dict[str, pd.Series]:
        if df is None or df.empty:
            return {}
        return {str(row["bvid"]): row for _, row in df.iterrows()}

    def _get_old_stats(self, bvid: str, pubdate_str: str) -> Optional[VideoStats]:
        if not self.use_old_data:
            return None

        if bvid in self._old_index:
            return VideoStats.from_dict(self._old_index[bvid].to_dict())

        # 检查是否为周期内新视频
        if self.old_time_threshold:
            try:
                pubdate = datetime.strptime(pubdate_str, "%Y-%m-%d %H:%M:%S")
                threshold = datetime.strptime(self.old_time_threshold, "%Y%m%d")
                if pubdate < threshold:
                    return None  # 早于周期，跳过
                return VideoStats()  # 新视频，返回空统计
            except:
                pass
        return None

    def _merge_collected(self, record: pd.Series, bvid: str) -> pd.Series:
        if bvid not in self._collected_index:
            return record

        collected = self._collected_index[bvid]
        record = record.copy()
        for field in COLLECTED_MERGE_FIELDS:
            if field in collected and pd.notna(collected[field]):
                record[field] = collected[field]
        return record

    def process(
        self,
        new_data: pd.DataFrame,
        old_data: Optional[pd.DataFrame] = None,
        collected_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """批量处理记录"""
        self._old_index = self._build_index(old_data) if self.use_old_data else {}
        self._collected_index = self._build_index(collected_data)

        results = []

        for _, record in new_data.iterrows():
            bvid = record.get("bvid")
            if not bvid:
                continue

            pubdate = record.get("pubdate", "")
            old_stats = self._get_old_stats(bvid, pubdate)

            # use_old_data 但找不到旧数据且不在周期内
            if self.use_old_data and old_stats is None and self.old_time_threshold:
                continue

            record = self._merge_collected(record, bvid)
            new_stats = VideoStats.from_dict(record.to_dict())

            try:
                score_result = calculate_score(new_stats, old_stats, self.ranking_type)
                results.append(_build_output_dict(record, score_result))
            except Exception as e:
                logger.error(f"处理失败 (bvid: {bvid}): {e}")

        return pd.DataFrame(results)


def process_records(
    new_data: pd.DataFrame,
    old_data: Optional[pd.DataFrame] = None,
    use_old_data: bool = False,
    collected_data: Optional[pd.DataFrame] = None,
    ranking_type: str = "daily",
    old_time_toll: Optional[str] = None,
) -> pd.DataFrame:
    """便捷函数"""
    type_map = {
        "daily": RankingType.DAILY,
        "weekly": RankingType.WEEKLY,
        "monthly": RankingType.MONTHLY,
        "annual": RankingType.ANNUAL,
        "special": RankingType.SPECIAL,
    }
    processor = RecordProcessor(
        ranking_type=type_map.get(ranking_type, RankingType.DAILY),
        use_old_data=use_old_data,
        old_time_threshold=old_time_toll,
    )
    return processor.process(new_data, old_data, collected_data)
