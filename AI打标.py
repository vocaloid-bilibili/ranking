# AI打标.py
import asyncio
from services.ai_tagger import AITagger
from common.config import get_app_config, get_paths, get_templates
from common.dates import get_daily_dates


async def main():
    dates = get_daily_dates()
    paths = get_paths()
    templates = get_templates()

    input_file = paths.daily_diff_new / templates.diff_new.format(**dates)
    output_file = (
        paths.daily_diff_new
        / f"新曲{dates['new_date']}与新曲{dates['old_date']}AI.xlsx"
    )

    tagger = AITagger(input_file=input_file, output_file=output_file)
    await tagger.run()


if __name__ == "__main__":
    asyncio.run(main())
