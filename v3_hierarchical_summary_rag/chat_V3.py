import sys
import os

# Bootstrap: Add V3 directory to path so stage-specific imports work out-of-the-box
v3_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(v3_dir)

from main_chat import main

if __name__ == "__main__":
    main()
