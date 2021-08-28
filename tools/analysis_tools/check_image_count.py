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
	    list = os.listdir(i)
	    number_files += len(list) / 2
	    print(i)
	    count += 1
	    print(count, len(subdirectories), number_files)

#print(number_files)

"""
data_root = '/home/max/server_1/home/d.grushevskaya1/projects/dron_maks/full_data/annotation/'
files = glob(data_root + '*.json')
files.drop('/home/max/server_1/home/d.grushevskaya1/projects/dron_maks/full_data/annotation/ch01_20200703110241-part 00000.json')
"""
