# video/card.py
"""成就卡片和头部渲染"""

from pathlib import Path
from typing import Dict, Optional
from io import BytesIO

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

from video.utils import wrap_text


class CardRenderer:
    """成就卡片渲染器"""

    def __init__(
        self,
        *,
        cache_dir: Path,
        font_regular: str,
        font_bold: str,
        card_width: int,
        card_height: int,
        card_radius: int,
    ):
        self.cache_dir = cache_dir
        self.font_file = font_regular
        self.font_bold_file = font_bold
        self.card_w = card_width
        self.card_h = card_height
        self.card_radius = card_radius

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cover_image(self, url: str, bvid: str) -> Image.Image:
        """获取封面图片（带缓存）"""
        cover_cache = self.cache_dir / bvid / "cover.jpg"

        if cover_cache.exists():
            try:
                return Image.open(cover_cache).convert("RGBA")
            except:
                pass

        if url and url.startswith("http"):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGBA")
                    cover_cache.parent.mkdir(parents=True, exist_ok=True)
                    img.convert("RGB").save(cover_cache)
                    return img
            except:
                pass

        return Image.new("RGBA", (300, 200), (200, 200, 200, 255))

    def create_card(self, row: pd.Series) -> Image.Image:
        """创建成就卡片"""
        title = str(row.get("title", ""))
        bvid = str(row.get("bvid", ""))
        author = str(row.get("author", ""))
        pubdate = str(row.get("pubdate", ""))
        image_url = str(row.get("image_url", ""))
        crossed_val = int(row.get("10w_crossed", 0))
        achievement_text = f"{crossed_val * 10}万播放达成!!"

        img = Image.new("RGBA", (self.card_w, self.card_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # 背景
        draw.rounded_rectangle(
            (0, 0, self.card_w, self.card_h),
            radius=self.card_radius,
            fill=(255, 255, 255, 255),
        )

        # 封面
        margin = 20
        cover_h = self.card_h - 2 * margin
        cover_w = int(cover_h * (16 / 9))

        cover_img = self._get_cover_image(image_url, bvid)
        cover_img = cover_img.resize((cover_w, cover_h), Image.Resampling.LANCZOS)

        mask = Image.new("L", (cover_w, cover_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, cover_w, cover_h), radius=15, fill=255
        )
        img.paste(cover_img, (margin, margin), mask)

        # 文字区域
        text_x = margin + cover_w + 30
        text_y = margin + 5
        text_width = self.card_w - text_x - margin

        try:
            f_title = ImageFont.truetype(self.font_bold_file, 34)
            f_info = ImageFont.truetype(self.font_file, 22)
            f_author = ImageFont.truetype(self.font_bold_file, 28)
            f_achieve = ImageFont.truetype(self.font_bold_file, 54)
        except:
            f_title = f_info = f_author = f_achieve = ImageFont.load_default()

        # 标题
        lines = wrap_text(title, f_title, text_width)
        for line in lines[:2]:
            draw.text((text_x, text_y), line, font=f_title, fill=(20, 20, 20))
            text_y += 45

        # 信息
        text_y += 5
        draw.text(
            (text_x, text_y),
            f"{bvid}  {pubdate}",
            font=f_info,
            fill=(100, 100, 100),
        )
        text_y += 32
        draw.text(
            (text_x, text_y),
            f"作者：{author}",
            font=f_author,
            fill=(60, 60, 60),
        )

        # 成就文字
        achieve_y = self.card_h - margin - 60
        draw.text(
            (text_x + 2, achieve_y + 2),
            achievement_text,
            font=f_achieve,
            fill="#CCAC00",
        )
        draw.text(
            (text_x, achieve_y),
            achievement_text,
            font=f_achieve,
            fill="#FFD700",
        )

        return img

    def create_header(
        self,
        width: int,
        height: int,
        opacity: float,
        ed_info: Optional[Dict] = None,
        list_top_y: Optional[int] = None,
    ) -> Image.Image:
        """创建成就页面头部"""
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        alpha = int(255 * opacity)
        if alpha <= 0:
            return img

        try:
            title_font = ImageFont.truetype(self.font_bold_file, 80)
        except:
            title_font = ImageFont.load_default()

        # 标题
        title = "今日成就达成"
        bbox = title_font.getbbox(title)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = (width - tw) // 2, 150

        draw.text((tx + 3, ty + 3), title, font=title_font, fill=(255, 255, 255, alpha))
        draw.text((tx, ty), title, font=title_font, fill=(0, 139, 139, alpha))

        # ED 信息
        if ed_info and (ed_info.get("name") or ed_info.get("bvid")):
            line1 = f"ED：{ed_info.get('name', '')}"
            if ed_info.get("author"):
                line1 += f" / {ed_info['author']}"
            line2 = ed_info.get("bvid", "")

            try:
                ed_font = ImageFont.truetype(self.font_file, 32)
            except:
                ed_font = ImageFont.load_default()

            region_top = ty + th + 25
            region_bottom = list_top_y if list_top_y is not None else int(height * 0.5)
            region_bottom = max(0, min(height, region_bottom))

            block_h = 80
            if region_bottom - region_top >= block_h + 10:
                ed_y = region_top + (region_bottom - region_top - block_h) // 2
            else:
                ed_y = region_top

            l1_box = ed_font.getbbox(line1)
            l2_box = ed_font.getbbox(line2)
            w_blk = max(l1_box[2], l2_box[2])
            ed_x = width - 80 - w_blk

            draw.text(
                (ed_x + 2, ed_y + 2), line1, font=ed_font, fill=(255, 255, 255, alpha)
            )
            draw.text((ed_x, ed_y), line1, font=ed_font, fill=(0, 0, 0, alpha))

            y2 = ed_y + 40
            draw.text(
                (ed_x + 2, y2 + 2), line2, font=ed_font, fill=(255, 255, 255, alpha)
            )
            draw.text((ed_x, y2), line2, font=ed_font, fill=(0, 0, 0, alpha))

        return img
