# utils/calculator.py
# 计算模块: 处理视频数据的计算逻辑
# 包括播放、收藏、硬币、点赞等数据的分数计算和排名更新

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
from math import ceil, floor
from utils.io_utils import format_columns

def calculate_scores(view: int, favorite: int, coin: int, like: int, danmaku: int, reply: int, share: int, copyright: int, ranking_type: str):
    """
    计算视频的各项评分
    
    Args:
        view: 播放
        favorite: 收藏
        coin: 硬币
        like: 点赞
        copyright: 版权类型(1,101为自制,2为转载)
        ranking_type: 榜单类型（'daily', 'weekly', 'monthly', 'annual', 'special'）。
    
    Returns:
        tuple: (播放分,收藏分,硬币分,点赞分,修正系数A,修正系数B,修正系数C)
    """
    viewR, favoriteR, coinR, likeR, danmakuR, replyR, shareR, fixA, fixB, fixC, fixD = (0.0,) * 11
    # 版权判定: 自制=1, 转载=2
    copyright = 1 if copyright in [1, 3, 101] else 2
    # 特殊情况处理: 如果有其他互动但没有投币,虚设为1参与计算
    coin = 1 if (coin == 0 and view > 0 and favorite > 0 and like > 0) else coin  
    # 计算修正系数A(搬运稿硬币得分补偿)
    fixA = 0 if coin <= 0 else (1 if copyright == 1 else ceil(max(1, (view + 20 * favorite + 40 * coin + 10 * like) / (200 * coin)) * 100) / 100)  
    
    # 计算修正系数B(云视听小电视等高播放收藏、低硬币点赞抑制系数)
    fixB = 0 if view + 20 * favorite <= 0 else ceil(min(1, 3 * max(0, (20 * coin * fixA + 10 * like)) / (view + 20 * favorite)) * 100) / 100

    # 计算修正系数C(梗曲等高点赞、低收藏抑制系数)
    fixC = 0 if like + favorite <= 0 else ceil(min(1, (like + favorite + 20 * coin * fixA)/(2 * like + 2 * favorite)) * 100) / 100

    # 日刊/周刊评分计算
    if ranking_type in ('daily', 'weekly'):
        viewR = 0 if view <= 0 else max(ceil(min(max((fixA * coin + favorite), 0) * 10 / view, 1) * 100) / 100, 0)
        favoriteR = 0 if favorite <= 0 else max(ceil(min((favorite + 2 * fixA * coin) * 10 / (favorite * 10 + view) * 20, 20) * 100) / 100, 0)
        coinR = 0 if fixA * coin * 40 + view <= 0 else max(ceil(min((fixA * coin * 40) / (fixA * coin * 20 + view) * 40, 40) * 100) / 100, 0)
        likeR = 0 if like <= 0 else max(floor(min(5, max(fixA * coin + favorite, 0) / (like * 20 + view) * 100) * 100) / 100, 0)
    # 月刊/年刊/特刊评分计算
    elif ranking_type in ('monthly', 'annual', 'special'):
        viewR = 0 if view <= 0 else max(ceil(min(max((fixA * coin + favorite), 0) * 15 / view, 1) * 100) / 100, 0)
        favoriteR = 0 if favorite <= 0 else max(ceil(min((favorite + 2 * fixA * coin) * 10 / (favorite * 10 + view) * 20, 20) * 100) / 100, 0)
        coinR = 0 if fixA * coin * 40 + view <= 0 else max(ceil(min((fixA * coin * 40) / (fixA * coin * 20 + view) * 40, 40) * 100) / 100, 0)
        likeR = 0 if like <= 0 else max(floor(min(5, max(fixA * coin + favorite, 0) / (like * 20 + view) * 100) * 100) / 100, 0)
    if ranking_type in ('annual'):
        viewR = viewR / 2 + 0.5
        favoriteR = favoriteR / 2 + 10
        coinR = coinR / 2 + 20
        likeR = likeR / 2 + 2.5
    
    return viewR, favoriteR, coinR, likeR, danmakuR, replyR, shareR, fixA, fixB, fixC, fixD

def calculate_scores_v2(view: int, favorite: int, coin: int, like: int, danmaku: int, reply: int, share: int, copyright: int, ranking_type: str):
    """
    计算视频的各项评分
    
    Args:
        view: 播放
        favorite: 收藏
        coin: 硬币
        like: 点赞
        copyright: 版权类型(1,101为自制,2为转载)
        ranking_type: 榜单类型（'daily', 'weekly', 'monthly', 'annual', 'special'）。
    
    Returns:
        tuple: (播放分,收藏分,硬币分,点赞分,修正系数A,修正系数B,修正系数C,修正系数D)
    """
    viewR, favoriteR, coinR, likeR, danmakuR, replyR, shareR, fixA, fixB, fixC, fixD = (0.0,) * 11
    # 版权判定: 自制=1, 转载=2
    copyright = 1 if copyright in [1, 3, 101] else 2
    # 特殊情况处理: 如果有其他互动但没有投币,虚设为1参与计算
    coin = 1 if (coin == 0 and view > 0 and favorite > 0 and like > 0) else coin  
    # 计算修正系数A(搬运稿硬币得分补偿)
    fixA = 0 if coin <= 0 else (1 if copyright == 1 else ceil(max(1, (view + 40 * favorite + 10 * like) / (150 * coin + 50 * max(0, danmaku))) * 100) / 100)  
    
    # 计算修正系数B(云视听小电视等高播放收藏、低硬币点赞抑制系数)
    fixB = 0 if view + 20 * favorite <= 0 else ceil(min(1, 3 * max(0, (20 * coin * fixA + 10 * like)) / (view + 20 * favorite)) * 100) / 100

    # 计算修正系数C(梗曲等高点赞、低收藏抑制系数)
    fixC = 0 if like + favorite <= 0 else ceil(min(1, (like + favorite + 20 * coin * fixA)/(2 * like + 2 * favorite)) * 100) / 100

    # 计算修正系数D(评论异常视频抑制系数)
    fixD = 0 if reply <= 0 else ceil(min(1, max(1, favorite + like)/ (max(1, favorite + like) + 0.1 * reply)) ** 20 * 100) / 100
    # 日刊/周刊评分计算
    if ranking_type in ('daily', 'weekly'):
        viewR = 0 if view <= 0 else max(ceil(min(max((fixA * coin + favorite), 0) * 10 / view, 1) * 100) / 100, 0)
        favoriteR = 0 if favorite <= 0 else max(ceil(min((favorite + 2 * fixA * coin) * 10 / (favorite * 10 + view) * 20, 20) * 100) / 100, 0)
        coinR = 0 if fixA * coin * 40 + view <= 0 else max(ceil(min((fixA * coin * 40) / (fixA * coin * 20 + view) * 40, 40) * 100) / 100, 0)
        likeR = 0 if like <= 0 else max(floor(min(5, max(fixA * coin + favorite, 0) / (like * 20 + view) * 100) * 100) / 100, 0)
        danmakuR = 0 if danmaku <= 0 else max(ceil(min(100, max(0, (20 * max(0, reply) + favorite + like)) / max(1, danmaku, danmaku + reply)) * 100) / 100, 0)
        replyR = 0 if reply <= 0 else max(ceil(min((400 * reply + 10 * like + 10 * favorite) / (200 * reply + view) * 20, 40) * 100) / 100, 0)
        shareR = 0 if share <= 0 else max(ceil(min((2 * fixA * coin + favorite) / (5 * share + like) * 10, 10) * 100) / 100, 0)
    # 月刊/年刊/特刊评分计算
    elif ranking_type in ('monthly', 'annual', 'special'):
        viewR = 0 if view <= 0 else max(ceil(min(max((fixA * coin + favorite), 0) * 15 / view, 1) * 100) / 100, 0)
        favoriteR = 0 if favorite <= 0 else max(ceil(min((favorite + 2 * fixA * coin) * 10 / (favorite * 10 + view) * 20, 20) * 100) / 100, 0)
        coinR = 0 if fixA * coin * 40 + view <= 0 else max(ceil(min((fixA * coin * 40) / (fixA * coin * 20 + view) * 40, 40) * 100) / 100, 0)
        likeR = 0 if like <= 0 else max(ceil(min(5, max(fixA * coin + favorite, 0) / (like * 20 + view) * 100) * 100) / 100, 0)
        danmakuR = 0 if danmaku <= 0 else max(ceil(min(100, max(0, (20 * max(0, reply) + favorite + like)) / max(1, danmaku, danmaku + reply)) * 100) / 100, 0)
        replyR = 0 if reply <= 0 else max(ceil(min((400 * reply + 10 * like + 10 * favorite) / (200 * reply + view) * 20, 40) * 100) / 100, 0)
        shareR = 0 if share <= 0 else max(ceil(min((2 * fixA * coin + favorite) / (5 * share + like) * 10, 10) * 100) / 100, 0)
    if ranking_type in ('annual', 'special'):
        viewR = viewR / 2 + 0.5
        favoriteR = favoriteR / 2 + 10
        coinR = coinR / 2 + 20
        likeR = likeR / 2 + 2.5
        #danmakuR = danmakuR / 2 + 50
        replyR = replyR / 2 + 20
        shareR = shareR / 2 + 5
    return viewR, favoriteR, coinR, likeR, danmakuR, replyR, shareR, fixA, fixB, fixC, fixD

def calculate_points(diff: List[float], scores: Tuple[float, ...]) -> float:
    """
    根据数据增量和评分系数计算总分。
    
    Args:
        diff (list): 包含播放、收藏、硬币、点赞增量的列表。
        scores (tuple): 由 `calculate_scores` 返回的评分系数元组。

    Returns:
        float: 计算得到的总分。
    """
    # 处理特殊情况: 如果没有投币但有其他互动, 则将硬币虚设为1
    coin =  1 if (diff[2] == 0 and diff[0] > 0 and diff[1] > 0 and diff[3] > 0) else diff[2]
    
    # 计算各项分数
    #viewR, favoriteR, coinR, likeR, fixA = scores[:5]
    viewR, favoriteR, coinR, likeR, danmakuR, replyR, shareR, fixA, fixB, fixC, fixD = scores[:11]
    viewP = diff[0] * viewR             # 播放得分
    favoriteP = diff[1] * favoriteR     # 收藏得分
    coinP = coin * coinR * fixA         # 硬币得分
    likeP = diff[3] * likeR             # 点赞得分
    danmakuP = diff[4] * danmakuR       # 弹幕得分
    replyP = diff[5] * replyR * fixD    # 评论得分
    shareP = diff[6] * shareR           # 分享得分
    return viewP + favoriteP + coinP + likeP + danmakuP + replyP + shareP

def calculate_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """计算DataFrame中各项指标的排名。
    
    Args:
        df (pd.DataFrame): 待计算排名的DataFrame，

    Returns:
        pd.DataFrame: 增加了排名列（'rank', 'view_rank',等）并格式化后的DataFrame。
    """
    # 按总分（point）降序排序
    df = df.sort_values('point', ascending=False)
    # 分别计算单项排名
    for col in ['view', 'favorite', 'coin', 'like', 'danmaku', 'reply', 'share']:
        df[f'{col}_rank'] = df[col].rank(ascending=False, method='min')
    # 计算总排名
    df['rank'] = df['point'].rank(ascending=False, method='min')
    return format_columns(df)

def update_rank_and_rate(df_today: pd.DataFrame, prev_file_path: Path) -> pd.DataFrame:
    """与上期数据比较，更新排名变化和得分增长率。

    Args:
        df_today (pd.DataFrame): 当前周期的榜单数据。
        prev_file_path (str): 上一期榜单数据的文件路径。

    Returns:
        pd.DataFrame: 增加了'rank_before', 'point_before', 'rate'列的DataFrame。
    """
    df_prev = pd.read_excel(prev_file_path)
    prev_dict = df_prev.set_index('name')[['rank', 'point']].to_dict(orient='index')

    # 添加上期排名和分数
    df_today['rank_before'] = df_today['name'].map(lambda x: prev_dict.get(x, {}).get('rank', '-'))
    df_today['point_before'] = df_today['name'].map(lambda x: prev_dict.get(x, {}).get('point', '-'))

    # 计算增长率 = (当前分数 - 上期分数) / 上期分数
    df_today['rate'] = df_today.apply(
        lambda row: (
            'NEW' if row['point_before'] == '-' else
            'inf' if row['point_before'] == 0 else
            f"{(row['point'] - row['point_before']) / row['point_before']:.2%}"
        ), axis=1
    )
    df_today = df_today.sort_values('point', ascending=False)
    return df_today

def update_count(df_today: pd.DataFrame, prev_file_path: Path) -> pd.DataFrame:
    """更新视频的在榜次数。
    Args:
        df_today (pd.DataFrame): 当前周期的榜单数据。
        prev_file_path (str): 上一期榜单数据的文件路径。

    Returns:
        pd.DataFrame: 增加了'count'列或更新了该列的DataFrame。
    """
    df_prev = pd.read_excel(prev_file_path)
    # 读取上期榜单的在榜次数
    prev_count_dict = df_prev.set_index('name')['count'].to_dict()
    # 如果当前排名≤20则在榜次数+1
    df_today['count'] = df_today['name'].map(lambda x: prev_count_dict.get(x, 0)) + (df_today['rank'] <= 20).astype(int)
    return df_today

def calculate_differences(new: pd.Series, ranking_type: str, old: Optional[pd.Series] = None):
    """计算新旧数据之间的差值。
    Args:
        new (pd.Series): 新数据记录。
        ranking_type (str): 榜单类型。
        old (pd.Series, optional): 旧数据记录。如果为None，则差值等于新数据。

    Returns:
        dict: 一个包含各项数据差值的字典。
    """
    if ranking_type in ('daily', 'weekly', 'monthly', 'annual'):
        if old is None:
            raise
        return {col: new[col] - old.get(col, 0) for col in ['view', 'favorite', 'coin', 'like', 'danmaku', 'reply', 'share']}
    # 特刊按总数据值计算
    elif ranking_type == 'special':
        return {col: new[col] for col in ['view', 'favorite', 'coin', 'like', 'danmaku', 'reply', 'share']}
    else:
        raise

def calculate(new: pd.Series, old: Optional[pd.Series], ranking_type: str):
    """执行完整的单条记录评分计算流程。

    该流程包括：计算数据差值、计算各项评分系数、计算最终总分。

    Args:
        new (pd.Series): 新的视频数据记录。
        old (dict): 旧的视频数据记录。
        ranking_type (str): 榜单类型。

    Returns:
        list: 包含差值、评分系数和总分的计算结果列表。
    """
    diff = [calculate_differences(new, ranking_type, old)[col] for col in ['view', 'favorite', 'coin', 'like', 'danmaku', 'reply', 'share']]
    scores = calculate_scores_v2(diff[0], diff[1], diff[2], diff[3], diff[4], diff[5], diff[6], new['copyright'], ranking_type)
    point = round(scores[8] * scores[9] * calculate_points(diff, scores))
    
    return diff + list(scores) + [point]

def merge_duplicate_names(df: pd.DataFrame) -> pd.DataFrame:
    """合并DataFrame中具有相同曲名的重复记录。

    当同一个曲名有多条记录时（例如，不同UP主上传的同一首歌曲），
    此函数会根据'point'列的值，只保留得分最高的那条记录。

    Args:
        df (pd.DataFrame): 包含可能重复曲名记录的DataFrame。

    Returns:
        pd.DataFrame: 合并了重复记录后的DataFrame。
    """
    merged_df = pd.DataFrame()
    grouped = df.groupby('name')
      
    for _, group in grouped:
        if len(group) > 1:
            # 获取组内最高分的记录
            best_record = group.loc[group['point'].idxmax()].copy()
            # 将最高分记录添加到结果中
            merged_df = pd.concat([merged_df, pd.DataFrame([best_record])])
        else: 
            # 无重复则直接添加
            merged_df = pd.concat([merged_df, group])
    return merged_df

def calculate_threshold(current_streak: int, census_mode: bool, base_threshold: int, streak_threshold: int) -> int:
    """根据连续未达标次数和模式，计算播放增长阈值。

    Args:
        current_streak (int): 当前连续未达标次数。
        census_mode (bool): 是否为普查模式。
        base_threshold (int): 基础阈值。
        streak_threshold (int): 触发动态阈值的连续未达标次数界限。

    Returns:
        int: 计算得到的播放增长阈值。
    """
    # 如果不是普查模式，统一使用基础阈值
    if not census_mode:
        return base_threshold
    # 在普查模式下，根据超过阈值的天数动态调整阈值
    # 每多一天，阈值增加一个基数，但增长有上限（最多7倍）
    gap = min(7, max(0, current_streak - streak_threshold))
    return base_threshold * (gap + 1)

def calculate_failed_mask(all_songs_df: pd.DataFrame, update_df: pd.DataFrame, census_mode: bool, streak_threshold: int) -> pd.Series:
    """计算并返回一个布尔掩码，标记哪些视频应被视为失效。

    Args:
        all_songs_df (pd.DataFrame): 包含所有已收录歌曲的DataFrame。
        update_df (pd.DataFrame): 包含本轮已成功更新数据的DataFrame。
        census_mode (bool): 是否为普查模式。
        streak_threshold (int): 触发动态阈值的连续未达标次数界限。

    Returns:
        pd.Series: 一个布尔Series，True表示视频失效。
    """
    # 普查模式下，任何未在更新列表中的视频都视为失效
    if census_mode:
        return ~all_songs_df['bvid'].isin(update_df['bvid'])
    # 常规模式下，只有连续未达标次数低于阈值且未被更新的视频才被视为失效
    mask = (
        (all_songs_df['streak'] < streak_threshold) & 
        ~all_songs_df['bvid'].isin(update_df['bvid'])
    )
    return mask