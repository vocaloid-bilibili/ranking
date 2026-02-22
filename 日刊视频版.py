# 日刊视频版.py
"""日刊视频生成入口"""

from video.flow import DailyVideoFlow


if __name__ == "__main__":
    flow = DailyVideoFlow()
    flow.run()
    input("按回车键退出...")
