import mmcv
import numpy as np
import json
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):
    
    CLASSES = (['drone'])
    def load_annotations(self, ann_file):
        f = open(ann_file)
        data = json.load(f)
        data_infos = []
        for i in data:
            i['ann']['bboxes'] = np.array(i['ann']['bboxes']).astype(np.float32)
            #i['ann']['bboxes'] = np.array([[ (bboxes[0][0]-0.5*bboxes[0][2]), (bboxes[0][1]-0.5*bboxes[0][3]), (bboxes[0][0]+0.5*bboxes[0][2]), (bboxes[0][1]+0.5*bboxes[0][3]),]]).astype(np.float32)
            i['ann']['labels'] = np.array(i['ann']['labels']).astype(np.int64)
            data_infos.append(i)
        return data_infos

    def get_ann_infos(self, idx):
        return self.data_infos[idx]['ann']
