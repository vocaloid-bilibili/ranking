# utils/climax_clipper.py
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import librosa
import numpy as np
import torch
from common.logger import logger


def _normalize(x: np.ndarray) -> np.ndarray:
    """将数组线性归一化到 [0, 1] 区间。"""
    x = x.astype(float)
    min_v = np.percentile(x, 2)
    max_v = np.percentile(x, 98)
    x = np.clip(x, min_v, max_v)
    if max_v - min_v < 1e-8:
        return np.zeros_like(x)
    return (x - min_v) / (max_v - min_v)


def _separate_vocals_demucs(audio_path: str, sr: int) -> Optional[np.ndarray]:
    """
    使用Demucs分离人声。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_out = Path(temp_dir)

        device = "cpu"
        if torch and torch.cuda.is_available():
            device = "cuda"

        cmd = [
            "demucs",
            "-n",
            "htdemucs",
            "--two-stems=vocals",
            "-d",
            device,
            str(audio_path),
            "-o",
            str(temp_out),
        ]

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            logger.warning("Demucs 分离失败或未安装，将跳过人声过滤步骤。")
            return None

        found_files = list(temp_out.rglob("vocals.wav"))
        target_file = found_files[0] if found_files else None

        if target_file and target_file.exists():
            try:
                y_vocals, _ = librosa.load(str(target_file), sr=sr, mono=True)
                return y_vocals
            except Exception:
                return None
        return None


def _compute_block_chroma_repetition(
    y: np.ndarray, sr: int, block_sec: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算重复度。
    """
    cqt_hop = 1024
    try:
        chroma_full = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=cqt_hop)
    except Exception:
        n_dummy = int(len(y) / (sr * block_sec))
        return np.zeros(n_dummy), np.linspace(0, len(y) / sr, n_dummy)

    n_frames = chroma_full.shape[1]
    frames_per_sec = sr / cqt_hop
    frames_per_block = int(block_sec * frames_per_sec)
    if frames_per_block < 1:
        frames_per_block = 1

    n_blocks = n_frames // frames_per_block
    if n_blocks < 4:
        return np.zeros(n_blocks), np.linspace(
            block_sec / 2, n_blocks * block_sec - block_sec / 2, n_blocks
        )

    chroma_blocks = []
    for i in range(n_blocks):
        start_f = i * frames_per_block
        end_f = (i + 1) * frames_per_block
        block_chroma = chroma_full[:, start_f:end_f]
        if block_chroma.shape[1] > 0:
            chroma_mean = block_chroma.mean(axis=1)
            norm = np.linalg.norm(chroma_mean) + 1e-8
            chroma_blocks.append(chroma_mean / norm)
        else:
            chroma_blocks.append(np.zeros(12))

    chroma_blocks_arr = np.stack(chroma_blocks, axis=0)
    sim = chroma_blocks_arr @ chroma_blocks_arr.T
    rep_score = sim.sum(axis=1) - 1.0
    rep_score = _normalize(rep_score)
    block_times = (np.arange(len(rep_score)) + 0.5) * block_sec
    return rep_score, block_times


def find_climax_segment(
    audio_path: str, clip_duration: float = 20.0, hop_length: int = 512
) -> Tuple[float, float]:
    """高潮检测算法"""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        return 0.0, clip_duration

    # 1. 响度
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # 2. 重复度
    rep_block, block_times = _compute_block_chroma_repetition(y, sr, block_sec=1.0)

    # 3. 对齐数据长度
    min_len = len(rms)
    frame_idx = np.arange(min_len)
    frame_times = librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length)

    # 将块级重复度插值到帧级
    if len(rep_block) > 1:
        rep_frame = np.interp(frame_times, block_times, rep_block)
    else:
        rep_frame = np.zeros_like(rms)

    rms_n = _normalize(rms)
    rep_n = _normalize(rep_frame)

    raw_score = 0.65 * rms_n + 0.35 * rep_n

    y_vocals = _separate_vocals_demucs(audio_path, sr)

    if y_vocals is not None:
        y_vocals = y_vocals[: len(y)]

        vocal_rms = librosa.feature.rms(y=y_vocals, hop_length=hop_length)[0]
        vocal_rms = vocal_rms[:min_len]
        vocal_rms_n = _normalize(vocal_rms)

        silence_thresh = 0.15

        vocal_mask = np.where(vocal_rms_n > silence_thresh, 1.0, 0.0)

        if np.mean(vocal_mask) > 0.05:
            raw_score = raw_score * vocal_mask

    frames_per_sec = sr / hop_length
    window_frames = int(clip_duration * frames_per_sec)
    duration = len(y) / sr

    if window_frames <= 1 or window_frames >= len(raw_score):
        return 0.0, min(float(duration), clip_duration)

    kernel = np.ones(window_frames, dtype=float)
    window_scores = np.convolve(raw_score, kernel, mode="valid")

    start_frame_idx = np.arange(len(window_scores))
    start_times = librosa.frames_to_time(start_frame_idx, sr=sr, hop_length=hop_length)

    start_margin = 10.0
    end_margin = 10.0
    max_ratio = 0.60

    if duration <= clip_duration + start_margin + end_margin:
        min_start = 0.0
        max_start = max(0.0, duration - clip_duration)
    else:
        min_start = start_margin
        max_start_tail = duration - clip_duration - end_margin
        max_start_ratio = duration * max_ratio
        max_start = min(max_start_tail, max_start_ratio)
        max_start = max(min_start, max_start)

    valid_mask = (start_times >= min_start) & (start_times <= max_start)
    masked_scores = window_scores.copy()
    masked_scores[~valid_mask] = -1e18

    best_idx = int(np.argmax(masked_scores))
    rough_start = float(start_times[best_idx])
    rough_start = max(0.0, min(rough_start, max_start))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    search_range = 1.0

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    candidates = beat_times[
        (beat_times >= rough_start - search_range)
        & (beat_times <= rough_start + search_range)
    ]

    final_start = rough_start
    if len(candidates) > 0:
        final_start = float(candidates[np.argmin(np.abs(candidates - rough_start))])
    else:
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames"
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
        candidates = onset_times[
            (onset_times >= rough_start - search_range)
            & (onset_times <= rough_start + search_range)
        ]
        if len(candidates) > 0:
            final_start = float(candidates[np.argmin(np.abs(candidates - rough_start))])

    final_start = max(0.0, min(final_start, max_start))
    final_end = min(float(duration), final_start + clip_duration)

    return float(final_start), float(final_end)
