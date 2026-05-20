# scripts/_bootstrap.py
"""确保项目根目录在 sys.path 中，并切换 cwd 到项目根"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 让 config/app.yaml、data/ 等相对路径正常工作
os.chdir(_PROJECT_ROOT)
