# 重复检测.py
"""疑似重复曲目检测"""

from services.duplicate import run_duplicate_detection


if __name__ == "__main__":
    run_duplicate_detection()
    input("按回车键退出...")
