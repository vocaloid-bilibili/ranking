# 导出特殊JSON.py
from pathlib import Path
from services.json_export import export_special

if __name__ == "__main__":
    export_special(name="哈基米")
    input("按回车键退出...")
