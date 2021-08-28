import numpy as np
import json
from glob import glob
from pathlib import Path
import os

import pandas as pd

if __name__ == '__main__':
	p = Path('/home/max/server_1/home/d.grushevskaya1/projects/dron_maks/full_data/annotation/')
	subdirectories = [x for x in p.iterdir() if x.is_dir()]
	number_files = 0
	count = 0
	for i in subdirectories:
	    _list = os.listdir(i)
	    number_files += len(_list) / 2
	    print(i)
	    count += 1
	    print(count, len(subdirectories), number_files)
