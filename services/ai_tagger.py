# services/ai_tagger.py
"""AI自动标注服务"""

import asyncio
import json
import re
import yaml
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
from openai import AsyncOpenAI, Timeout

from common.logger import logger
from common.io import save_excel, load_excel
from common.config import get_paths, Paths


COLOR_YELLOW = "FFFF00"
COLOR_LIGHT_BLUE = "ADD8E6"


@dataclass
class AIConfig:
    """AI服务配置"""

    provider: str
    model_name: str
    api_url: str
    api_key: str
    batch_size: int = 15
    max_concurrency: int = 10
    max_attempts: int = 10

    @classmethod
    def from_yaml(cls, path: Path = None) -> "AIConfig":
        """从YAML文件加载配置"""
        if path is None:
            path = get_paths().ai_config

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        provider = cfg.get("PROVIDER", "API")
        provider_cfg = cfg.get(provider, {})

        return cls(
            provider=provider,
            model_name=provider_cfg.get("API_MODEL", ""),
            api_url=provider_cfg.get("API_URL", ""),
            api_key=provider_cfg.get("API_KEY", ""),
            batch_size=cfg.get("BATCH_SIZE", 15),
            max_concurrency=cfg.get("MAX_CONCURRENCY", 10),
            max_attempts=cfg.get("MAX_ATTEMPTS", 10),
        )


class AITagger:
    """使用AI对视频数据进行筛选和标注"""

    TAGGING_COLS = ["synthesizer", "vocal", "type"]
    SEND_COLS = ["title", "uploader", "intro", "copyright"]

    def __init__(
        self,
        input_file: Path,
        output_file: Path,
        collected_file: Path = None,
        config: Optional[AIConfig] = None,
        paths: Optional[Paths] = None,
    ):
        self.input_file = input_file
        self.output_file = output_file

        # 加载路径配置
        self._paths = paths or get_paths()
        self.collected_file = collected_file or self._paths.collected

        # 加载AI配置
        self.config = config or AIConfig.from_yaml(self._paths.ai_config)
        self._validate_config()

        # 初始化客户端
        self.client = AsyncOpenAI(
            base_url=self.config.api_url,
            api_key=self.config.api_key,
            timeout=Timeout(60.0),
        )
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)

        # 加载prompt模板
        self.prompt_template = self._load_prompt_template()

    def _validate_config(self):
        """验证配置"""
        if not self.config.model_name:
            raise ValueError("未定义 API_MODEL")
        if self.config.provider not in ("API", "LOCAL"):
            raise ValueError(f"不支持的提供商: {self.config.provider}")
        logger.info(f"AI Tagger 使用提供商: {self.config.provider}")

    def _load_prompt_template(self) -> str:
        """加载并格式化prompt模板"""
        synthesizers, vocals = self._load_known_tags()

        try:
            template = self._paths.prompt_template.read_text(encoding="utf-8")
            return template.format(
                known_synthesizers="\n".join(sorted(synthesizers)) or "无",
                known_vocals="\n".join(sorted(vocals)) or "无",
            )
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"加载Prompt模板失败: {e}")
            raise

    def _load_known_tags(self) -> Tuple[Set[str], Set[str]]:
        """从收录曲目加载已知标签"""
        if not self.collected_file.exists():
            logger.warning(f"收录曲目文件不存在: {self.collected_file}")
            return set(), set()

        df = load_excel(self.collected_file)

        def extract_tags(series: pd.Series) -> Set[str]:
            tags = set()
            for item in series.dropna().astype(str):
                tags.update(t.strip() for t in item.split("、") if t.strip())
            return tags

        synthesizers = (
            extract_tags(df["synthesizer"]) if "synthesizer" in df.columns else set()
        )
        vocals = extract_tags(df["vocal"]) if "vocal" in df.columns else set()

        return synthesizers, vocals

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """解析AI响应中的JSON"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        logger.error("无法解析AI响应中的JSON")
        return None

    async def _call_api(self, batch_data: List[Dict]) -> Optional[List[Dict]]:
        """调用AI API"""
        user_content = json.dumps({"videos": batch_data}, ensure_ascii=False)

        for attempt in range(self.config.max_attempts):
            try:
                async with self.semaphore:
                    params = {
                        "model": self.config.model_name,
                        "messages": [
                            {"role": "system", "content": self.prompt_template},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.2,
                        "stream": True,
                    }
                    if self.config.provider == "API":
                        params["response_format"] = {"type": "json_object"}

                    response = await self.client.chat.completions.create(**params)

                    chunks = []
                    async for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            chunks.append(chunk.choices[0].delta.content)

                    content = "".join(chunks)

                if not content:
                    logger.warning(f"尝试 {attempt + 1}: AI返回空内容")
                    continue

                parsed = self._parse_json_response(content)
                if not parsed:
                    logger.warning(f"尝试 {attempt + 1}: JSON解析失败")
                    continue

                results = parsed.get("results", [])
                if len(results) == len(batch_data):
                    return results

                logger.warning(f"尝试 {attempt + 1}: 结果数量不匹配")

            except Exception as e:
                logger.error(f"尝试 {attempt + 1} 失败: {e}")

        return None

    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
        """准备数据，区分已标注和待处理"""
        df = pd.read_excel(self.input_file)

        for col in self.TAGGING_COLS:
            if col not in df.columns:
                df[col] = pd.NA
            df[col] = df[col].astype(object)

        df["status"] = "check"
        row_styles = {}

        # 标记已完成的
        tagged_mask = df[self.TAGGING_COLS].notna().all(axis=1)
        done_indices = df[tagged_mask].index
        df.loc[done_indices, "status"] = "done"
        for idx in done_indices:
            row_styles[idx] = COLOR_LIGHT_BLUE

        to_process = df[~tagged_mask].copy()
        logger.info(
            f"总计 {len(df)} 首，已标注 {len(done_indices)} 首，待处理 {len(to_process)} 首"
        )

        return df, to_process, row_styles

    def _apply_results(
        self,
        df: pd.DataFrame,
        indices: pd.Index,
        results: List[Dict],
        styles: Dict[int, str],
    ):
        """应用AI结果到DataFrame"""
        for idx, res in zip(indices, results):
            if not isinstance(res, dict):
                logger.warning(f"索引 {idx} 结果无效: {res}")
                continue

            if res.get("include"):
                tags = res.get("tags", {})
                for key, value in tags.items():
                    if key in df.columns:
                        df.at[idx, key] = value
                styles[idx] = COLOR_YELLOW
                df.at[idx, "status"] = "auto"
            else:
                for col in self.TAGGING_COLS:
                    df.at[idx, col] = pd.NA
                df.at[idx, "status"] = "check"

    async def run(self):
        """执行AI标注流程"""
        try:
            df, to_process, styles = self._prepare_data()
        except FileNotFoundError:
            logger.error(f"输入文件不存在: {self.input_file}")
            return

        if to_process.empty:
            logger.info("没有需要处理的歌曲")
            if not self.output_file.exists():
                save_excel(df, self.output_file, row_styles=styles)
            return

        save_excel(df, self.output_file, row_styles=styles)

        # 分批处理
        chunks = [
            to_process.iloc[i : i + self.config.batch_size]
            for i in range(0, len(to_process), self.config.batch_size)
        ]
        total = len(chunks)
        logger.info(f"开始处理 {len(to_process)} 首，共 {total} 批")

        async def process_chunk(chunk: pd.DataFrame):
            records = chunk.reindex(columns=self.SEND_COLS).to_dict("records")
            results = await self._call_api(records)
            return chunk, results

        tasks = [process_chunk(chunk) for chunk in chunks]
        completed, failed = 0, 0

        for future in asyncio.as_completed(tasks):
            count = completed + failed + 1
            try:
                chunk, results = await future
                if results:
                    self._apply_results(df, chunk.index, results, styles)
                    completed += 1
                    logger.info(f"批次 {count}/{total} 成功")
                else:
                    failed += 1
                    logger.warning(f"批次 {count}/{total} 失败")
            except Exception as e:
                failed += 1
                logger.error(f"批次 {count}/{total} 异常: {e}")

            save_excel(df, self.output_file, row_styles=styles)

        logger.info(f"完成：成功 {completed}，失败 {failed}")
