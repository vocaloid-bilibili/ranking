# 先做一个普通的周刊，然后筛选出翻唱数据，用特刊做法处理一遍
import asyncio
from ranking.processor import RankingProcessor
import pandas as pd

name = '2026-04-04'


async def main():
    data = pd.read_excel(f'./data/special/data/{name}.xlsx')
    data = data[data['type'] == '翻唱']
    data.to_excel(f'./data/special/data/{name}.xlsx')
    
    processor = RankingProcessor(period="special")
    await processor.run(song_data=name)


if __name__ == "__main__":
    asyncio.run(main())
