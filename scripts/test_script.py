import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.framsticks_interface.interface import get_interface

config = load_config()
interface = get_interface(config)
for i in range(3):
    print(interface.get_random_genotype('f1'))