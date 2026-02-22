# ranking/calculator.py
"""纯评分计算模块"""

from math import ceil, floor
from dataclasses import dataclass
from typing import Tuple
from common.models import VideoStats, ScoreResult, RankingType


@dataclass
class RateCoefficients:
    """评分系数"""

    view: float = 0.0
    favorite: float = 0.0
    coin: float = 0.0
    like: float = 0.0
    danmaku: float = 0.0
    reply: float = 0.0
    share: float = 0.0


@dataclass
class FixCoefficients:
    """修正系数"""

    a: float = 0.0  # 搬运稿硬币补偿
    b: float = 0.0  # 高播放低互动抑制
    c: float = 0.0  # 高点赞低收藏抑制
    d: float = 0.0  # 评论异常抑制


def normalize_copyright(copyright: int) -> int:
    """版权标准化：自制=1，转载=2"""
    return 1 if copyright in [1, 3, 101] else 2


def adjust_coin(stats: VideoStats) -> int:
    """调整硬币数：有互动但无投币时虚设为1"""
    if stats.coin == 0 and stats.view > 0 and stats.favorite > 0 and stats.like > 0:
        return 1
    return stats.coin


def compute_fix_coefficients(stats: VideoStats) -> FixCoefficients:
    """计算修正系数"""
    copyright = normalize_copyright(stats.copyright)
    coin = adjust_coin(stats)

    # 修正系数A：搬运稿硬币补偿
    if coin <= 0:
        fix_a = 0.0
    elif copyright == 1:
        fix_a = 1.0
    else:
        denominator = 150 * coin + 50 * max(0, stats.danmaku)
        if denominator <= 0:
            fix_a = 1.0
        else:
            fix_a = (
                ceil(
                    max(
                        1,
                        (stats.view + 40 * stats.favorite + 10 * stats.like)
                        / denominator,
                    )
                    * 100
                )
                / 100
            )

    # 修正系数B：高播放收藏、低硬币点赞抑制
    denominator_b = stats.view + 20 * stats.favorite
    if denominator_b <= 0:
        fix_b = 0.0
    else:
        fix_b = (
            ceil(
                min(1, 3 * max(0, 20 * coin * fix_a + 10 * stats.like) / denominator_b)
                * 100
            )
            / 100
        )

    # 修正系数C：高点赞低收藏抑制
    denominator_c = stats.like + stats.favorite
    if denominator_c <= 0:
        fix_c = 0.0
    else:
        fix_c = (
            ceil(
                min(
                    1,
                    (stats.like + stats.favorite + 20 * coin * fix_a)
                    / (2 * denominator_c),
                )
                * 100
            )
            / 100
        )

    # 修正系数D：评论异常抑制
    if stats.reply <= 0:
        fix_d = 0.0
    else:
        base = max(1, stats.favorite + stats.like)
        fix_d = ceil(min(1, base / (base + 0.1 * stats.reply)) ** 20 * 100) / 100

    return FixCoefficients(a=fix_a, b=fix_b, c=fix_c, d=fix_d)


def compute_rate_coefficients(
    stats: VideoStats, fix: FixCoefficients, ranking_type: RankingType
) -> RateCoefficients:
    """计算评分系数"""
    coin = adjust_coin(stats)

    # 日刊/周刊
    if ranking_type in (RankingType.DAILY, RankingType.WEEKLY):
        view_r = _calc_view_rate_short(stats, fix, coin)
        favorite_r = _calc_favorite_rate(stats, fix, coin)
        coin_r = _calc_coin_rate(stats, fix, coin)
        like_r = _calc_like_rate(stats, fix, coin)
        danmaku_r = _calc_danmaku_rate(stats)
        reply_r = _calc_reply_rate(stats)
        share_r = _calc_share_rate(stats, fix, coin)
    # 月刊/年刊/特刊
    else:
        view_r = _calc_view_rate_long(stats, fix, coin)
        favorite_r = _calc_favorite_rate(stats, fix, coin)
        coin_r = _calc_coin_rate(stats, fix, coin)
        like_r = _calc_like_rate(stats, fix, coin)
        danmaku_r = _calc_danmaku_rate(stats)
        reply_r = _calc_reply_rate(stats)
        share_r = _calc_share_rate(stats, fix, coin)

    # 年刊/特刊额外调整
    if ranking_type in (RankingType.ANNUAL, RankingType.SPECIAL):
        view_r = view_r / 2 + 0.5
        favorite_r = favorite_r / 2 + 10
        coin_r = coin_r / 2 + 20
        like_r = like_r / 2 + 2.5
        reply_r = reply_r / 2 + 20
        share_r = share_r / 2 + 5

    return RateCoefficients(
        view=view_r,
        favorite=favorite_r,
        coin=coin_r,
        like=like_r,
        danmaku=danmaku_r,
        reply=reply_r,
        share=share_r,
    )


def _calc_view_rate_short(stats: VideoStats, fix: FixCoefficients, coin: int) -> float:
    if stats.view <= 0:
        return 0.0
    return max(
        ceil(min(max(fix.a * coin + stats.favorite, 0) * 10 / stats.view, 1) * 100)
        / 100,
        0,
    )


def _calc_view_rate_long(stats: VideoStats, fix: FixCoefficients, coin: int) -> float:
    if stats.view <= 0:
        return 0.0
    return max(
        ceil(min(max(fix.a * coin + stats.favorite, 0) * 15 / stats.view, 1) * 100)
        / 100,
        0,
    )


def _calc_favorite_rate(stats: VideoStats, fix: FixCoefficients, coin: int) -> float:
    if stats.favorite <= 0:
        return 0.0
    return max(
        ceil(
            min(
                (stats.favorite + 2 * fix.a * coin)
                * 10
                / (stats.favorite * 10 + stats.view)
                * 20,
                20,
            )
            * 100
        )
        / 100,
        0,
    )


def _calc_coin_rate(stats: VideoStats, fix: FixCoefficients, coin: int) -> float:
    denom = fix.a * coin * 40 + stats.view
    if denom <= 0:
        return 0.0
    return max(
        ceil(min((fix.a * coin * 40) / (fix.a * coin * 20 + stats.view) * 40, 40) * 100)
        / 100,
        0,
    )


def _calc_like_rate(stats: VideoStats, fix: FixCoefficients, coin: int) -> float:
    if stats.like <= 0:
        return 0.0
    return max(
        ceil(
            min(
                5,
                max(fix.a * coin + stats.favorite, 0)
                / (stats.like * 20 + stats.view)
                * 100,
            )
            * 100
        )
        / 100,
        0,
    )


def _calc_danmaku_rate(stats: VideoStats) -> float:
    if stats.danmaku <= 0:
        return 0.0
    denom = max(1, stats.danmaku, stats.danmaku + stats.reply)
    return max(
        ceil(
            min(
                100,
                max(0, 20 * max(0, stats.reply) + stats.favorite + stats.like) / denom,
            )
            * 100
        )
        / 100,
        0,
    )


def _calc_reply_rate(stats: VideoStats) -> float:
    if stats.reply <= 0:
        return 0.0
    return max(
        ceil(
            min(
                (400 * stats.reply + 10 * stats.like + 10 * stats.favorite)
                / (200 * stats.reply + stats.view)
                * 20,
                40,
            )
            * 100
        )
        / 100,
        0,
    )


def _calc_share_rate(stats: VideoStats, fix: FixCoefficients, coin: int) -> float:
    if stats.share <= 0:
        return 0.0
    return max(
        ceil(
            min(
                (2 * fix.a * coin + stats.favorite)
                / (5 * stats.share + stats.like)
                * 10,
                10,
            )
            * 100
        )
        / 100,
        0,
    )


def compute_total_points(
    diff: VideoStats, rates: RateCoefficients, fix: FixCoefficients
) -> float:
    """计算总分"""
    coin = adjust_coin(diff)

    view_p = diff.view * rates.view
    favorite_p = diff.favorite * rates.favorite
    coin_p = coin * rates.coin * fix.a
    like_p = diff.like * rates.like
    danmaku_p = diff.danmaku * rates.danmaku
    reply_p = diff.reply * rates.reply * fix.d
    share_p = diff.share * rates.share

    return view_p + favorite_p + coin_p + like_p + danmaku_p + reply_p + share_p


def calculate_score(
    new_stats: VideoStats, old_stats: VideoStats | None, ranking_type: RankingType
) -> ScoreResult:
    """完整评分计算流程"""
    # 计算增量
    if old_stats is None:
        diff = new_stats
    else:
        diff = VideoStats(
            view=new_stats.view - old_stats.view,
            favorite=new_stats.favorite - old_stats.favorite,
            coin=new_stats.coin - old_stats.coin,
            like=new_stats.like - old_stats.like,
            danmaku=new_stats.danmaku - old_stats.danmaku,
            reply=new_stats.reply - old_stats.reply,
            share=new_stats.share - old_stats.share,
            copyright=new_stats.copyright,
        )

    # 计算系数
    fix = compute_fix_coefficients(diff)
    rates = compute_rate_coefficients(diff, fix, ranking_type)

    # 计算总分
    raw_point = compute_total_points(diff, rates, fix)
    point = round(fix.b * fix.c * raw_point)

    return ScoreResult(
        view=diff.view,
        favorite=diff.favorite,
        coin=diff.coin,
        like=diff.like,
        danmaku=diff.danmaku,
        reply=diff.reply,
        share=diff.share,
        view_rate=rates.view,
        favorite_rate=rates.favorite,
        coin_rate=rates.coin,
        like_rate=rates.like,
        danmaku_rate=rates.danmaku,
        reply_rate=rates.reply,
        share_rate=rates.share,
        fix_a=fix.a,
        fix_b=fix.b,
        fix_c=fix.c,
        fix_d=fix.d,
        point=point,
    )
