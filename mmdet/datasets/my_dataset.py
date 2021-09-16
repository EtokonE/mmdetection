import mmcv
import numpy as np
import json
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MyDataset(CustomDataset):
    '''Датасет содержит все картинки, прописанные в аннотациях'''
    CLASSES = (['drone'])
    def load_annotations(self, ann_file):
        f = open(ann_file)
        data = json.load(f)
        data_infos = []
        for i in data:
            i['ann']['bboxes'] = np.array(i['ann']['bboxes']).astype(np.float32)
            i['ann']['labels'] = np.array(i['ann']['labels']).astype(np.int64)
            data_infos.append(i)
        return data_infos

    def get_ann_infos(self, idx):
        return self.data_infos[idx]['ann']


@DATASETS.register_module()
class MyDataset_drop10(CustomDataset):
    '''Из датасета удалены все боксы, площадь которых меньше 10pix'''
    CLASSES = (['drone'])
    def load_annotations(self, ann_file):

        f = open(ann_file)
        data = json.load(f)

        data_infos = []
        for i in data:
                
            # Переменная для хранения боксов подлежащих удалению
            pop_list = []

            # Для каждого изображения необходимо проверить площадь каждого бокса
            for num_bbox in range(len(i['ann']['bboxes'])):
                if abs( (i['ann']['bboxes'][num_bbox][0] - i['ann']['bboxes'][num_bbox][2]) * (i['ann']['bboxes'][num_bbox][1] - i['ann']['bboxes'][num_bbox][3]) ) < 10:
                    pop_list.append(num_bbox)

            # Удаляем слишком маленькие боксы и метки их классов
            for el in pop_list[::-1]:
                i['ann']['bboxes'].pop(el)
                i['ann']['labels'].pop(el)
        
            i['ann']['bboxes'] = np.array(i['ann']['bboxes']).astype(np.float32)
            i['ann']['labels'] = np.array(i['ann']['labels']).astype(np.int64)
            
            if len(i['ann']['bboxes']) > 0:
                data_infos.append(i)

        return data_infos

    def get_ann_infos(self, idx):
        return self.data_infos[idx]['ann']


@DATASETS.register_module()
class MyDataset_ignore10(CustomDataset):
    '''В датасете игнорируются все боксы, площадь которых меньше 10pix'''
    CLASSES = (['drone'])
    def load_annotations(self, ann_file):

        f = open(ann_file)
        data = json.load(f)

        data_infos = []

        for i in data:
            # Боксы, которые будем игнорировать 
            bboxes_ignore = []
            labels_ignore = []
            # Для каждого изображения необходимо проверить площадь каждого бокса
            for num_bbox in range(len(i['ann']['bboxes'])):
                if abs( (i['ann']['bboxes'][num_bbox][0] - i['ann']['bboxes'][num_bbox][2]) * (i['ann']['bboxes'][num_bbox][1] - i['ann']['bboxes'][num_bbox][3]) ) < 10:    
                    bboxes_ignore.append(i['ann']['bboxes'][num_bbox])
                    labels_ignore.append(i['ann']['labels'][num_bbox])
                 
            # Загрузка аннотаций
            i['ann']['bboxes'] = np.array(i['ann']['bboxes']).astype(np.float32)
            i['ann']['labels'] = np.array(i['ann']['labels']).astype(np.int64)
            # Добавление в аннотацию боксов, которые будут проигнорированы
            if len(bboxes_ignore) > 0:
                i['ann']['bboxes_ignore'] = np.array(bboxes_ignore).astype(np.float32)
                i['ann']['labels_ignore'] = np.array(labels_ignore).astype(np.int64)
    
            data_infos.append(i)

        return data_infos

    def get_ann_infos(self, idx):
        return self.data_infos[idx]['ann']

