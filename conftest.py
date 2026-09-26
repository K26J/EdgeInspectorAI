import sys
import os

# 1. Dynamically calculate the absolute path of the folder containing this file
project_root = os.path.abspath(os.path.dirname(__file__))

# 2. Forcefully insert it at the very top of Python's internal search map
sys.path.insert(0, project_root)

print(f"\n[DEBUG] Forcing Python to recognize root directory: {project_root}")