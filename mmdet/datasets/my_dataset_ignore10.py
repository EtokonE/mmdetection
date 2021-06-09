import mmcv
import numpy as np
import json
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset_ignore10(CustomDataset):
    
    CLASSES = (['drone'])
    def load_annotations(self, ann_file):

        f = open(ann_file)
        data = json.load(f)

        data_infos = []

        for i in data:
        # Переменная для хранения боксов подлежащих удалению        
        bboxes_ignore = []        
        labels_ignore = []        
        # Для каждого изображения необходимо проверить площадь каждого бокса        
        for num_bbox in range(len(i['ann']['bboxes'])):            
        if abs( (i['ann']['bboxes'][num_bbox][0] - i['ann']['bboxes'][num_bbox][2]) * (i['ann']['bboxes'][num_bbox][1] - i['ann']['bboxes'][num_bbox][3]) ) < 10:                    
                bboxes_ignore.append(i['ann']['bboxes'][num_bbox])
                labels_ignore.append(i['ann']['labels'][num_bbox])
        

        i['ann']['bboxes'] = np.array(i['ann']['bboxes']).astype(np.float32)
        i['ann']['labels'] = np.array(i['ann']['labels']).astype(np.int64)
        if len(bboxes_ignore) > 0:
            i['ann']['bboxes_ignore'] = np.array(bboxes_ignore).astype(np.float32)            
            i['ann']['labels_ignore'] = np.array(labels_ignore).astype(np.int64)    
           
            
        data_infos.append(i)

        return data_infos

    def get_ann_infos(self, idx):
        return self.data_infos[idx]['ann']
