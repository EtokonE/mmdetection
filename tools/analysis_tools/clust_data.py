import numpy as np
import json
from glob import glob
import pandas as pd

data_root = '/home/max/server_1/home/d.grushevskaya1/projects/dron_maks/full_data/annotation/'
files = glob(data_root + '*.json')
bboxes = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        
        for image in data:
            for num_bbox in range(len(image['ann']['bboxes'])):
                rect_sides = [
                    abs(image['ann']['bboxes'][num_bbox][0] - image['ann']['bboxes'][num_bbox][2]),
                    abs(image['ann']['bboxes'][num_bbox][1] - image['ann']['bboxes'][num_bbox][3])
                ]

                bboxes.append(rect_sides)


df = pd.DataFrame(bboxes, columns=['x', 'y'])
df.to_csv('./clust.csv', index=False)
