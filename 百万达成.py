# 百万达成.py
from services.milestone import run_milestone_check

if __name__ == "__main__":
    run_milestone_check(save_weekly_only=True)
    input("按回车键退出...")
