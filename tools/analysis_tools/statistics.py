import pickle
import numpy as np
import json
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_iou(bb1, bb2):
    """
    Рассчитывает Intersection over Union (IoU) между двумя bounding boxes.

    Parameters
    ----------
    bb1 : list
        [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : list
        [x1, y1, x2, y2]
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert len(bb1) == 4 and len(bb2) == 4
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def statistic_2(result_pkl, annotation_json, out, iou_thr):
    """
    Соединяет файл аннотаций и результатов работы детектора в один json файл
    добавляя при этом необходимые статистики
    """
    print('Создаю файл')
    # Загружаем файл, созданный при помощи tools/test.py
    with open(str(result_pkl), 'rb') as pkl:
        data_pkl = pickle.load(pkl)

    # Загружаем файл аннотаций
    with open(str(annotation_json)) as js:
        data_js = json.load(js)
        assert len(data_pkl) == len(data_js)
        print('Вычисляю метрики')
        data_infos = []
        for i in range(len(data_js)):

            # Координаты, площадь сгенерированных и IOU боксов прошедших через порог iou_thr
            bbox_area = {}
            bbox_area['bboxes'] = []
            bbox_area['areas'] = []
            bbox_area['IOU'] = []

            # Площадь ground_truth боксов
            data_js[i]['area'] = []
            for num_bbox in range(len(data_js[i]['ann']['bboxes'])):
                data_js[i]['area'].append(
                    abs((data_js[i]['ann']['bboxes'][num_bbox][0] - data_js[i]['ann']['bboxes'][num_bbox][2]) * (
                                data_js[i]['ann']['bboxes'][num_bbox][1] - data_js[i]['ann']['bboxes'][num_bbox][3])))

            for j in range(20):
                # Проверяем порог истинности бокса
                if data_pkl[i][0][j][4] >= iou_thr:
                    # координаты сгенерированных боксов
                    bbox_area['bboxes'].append(data_pkl[i][0][j][:-1].tolist())
                    # Площадь сгенерированных боксов
                    bbox_area['areas'].append(str(abs(
                        (data_pkl[i][0][j][0] - data_pkl[i][0][j][2]) * (data_pkl[i][0][j][1] - data_pkl[i][0][j][3]))))
                    # IOU сгенерированных боксов
                    bbox_area['IOU'].append(str(round(data_pkl[i][0][j][4], 2)))
            data_js[i]['generated_bbox'] = bbox_area

            # Определяем истинность боксов
            bb1 = data_js[i]['ann']['bboxes']
            bb2 = data_js[i]['generated_bbox']['bboxes']

            # Максимальный IOU предсказанного бокса с одним из истинных
            ious = []
            for pred in bb2:
                temp = 0
                for gt in bb1:
                    calculate_iou = get_iou(gt, pred)
                    if calculate_iou >= temp:
                        temp = calculate_iou
                ious.append(temp)
            data_js[i]['generated_bbox']['iou_predicted_to_gt'] = ious

            # Максимальный IOU истинного бокса с одним из предсказанных
            ious = []
            for gt in bb1:
                temp = 0
                for pred in bb2:
                    calculate_iou = get_iou(gt, pred)
                    if calculate_iou >= temp:
                        temp = calculate_iou
                ious.append(temp)
            data_js[i]['generated_bbox']['iou_gt_to_predicted'] = ious
    # Пишем результаты в json файл
    with open(str(out), 'w') as out_file:
        json.dump(data_js, out_file)
  

def plot_graphs(statistic_json, iou_thr, ignore_thr, graph_scale, work_dir):
    """
    Парсит файл со статистиками и отрисовывает необходимые графики
    """
    print('Рисую графики')
    with open(str(statistic_json)) as st:
        stats = json.load(st)
        # Площадь всех ground_truth боксов
        gt_areas = [image['area'][index_bbox] for image in stats for index_bbox in range(len(image['area']))]
        df_gt_areas = pd.DataFrame({'gt_areas': gt_areas})

        # Площадь отрисованных боксов
        pred = [image.get('generated_bbox') for image in stats]
        positive_bbox_area = []
        negative_bbox_area = []
        unidentified = []
        for predicted in pred:
            for i in range(len(predicted.get('IOU'))):
                # Площадь правильно задетектированных боксов
                if float(predicted.get('iou_predicted_to_gt')[i]) > iou_thr:
                    positive_bbox_area.append(float(predicted.get('areas')[i]))
                # Площадь неправильно задетектированных боксов
                else:
                    negative_bbox_area.append(float(predicted.get('areas')[i]))

        # Площадь истинных боксов, которые быди проигнорированны детектором
        info = [image for image in stats]
        for predicted in info:
            for i in range(len(predicted.get('generated_bbox').get('iou_gt_to_predicted'))):
                if float(predicted.get('generated_bbox').get('iou_gt_to_predicted')[i]) < ignore_thr:
                    unidentified.append(float(predicted.get('area')[i]))

        df_positive = pd.DataFrame({'positive': positive_bbox_area})
        df_negative = pd.DataFrame({'negative': negative_bbox_area})
        df_unidentified = pd.DataFrame({'unidentified': unidentified})

    # _________________График_правильно_отрисованных_боксов_______________
    gt_area_plot = df_positive.plot.hist(bins=200,
                                         log=True,
                                         figsize=(18, 9),
                                         xlim=(0, graph_scale)
                                         )
    gt_area_plot.set_title('Распределения площади правильно предсказанных боксов', weight='bold', size=20)
    gt_area_plot.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    gt_area_plot.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    gt_area_plot.figure.savefig(os.path.join(work_dir, 'positive_area.pdf'))

    # _________________График_ошибочно_предсказанных_боксов________________
    gt_area_plot = df_negative.plot.hist(bins=200,
                                         log=True,
                                         figsize=(18, 9),
                                         xlim=(0, graph_scale))
    gt_area_plot.set_title('Распределения площади ошибочно предсказанных боксов', weight='bold', size=20)
    gt_area_plot.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    gt_area_plot.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    gt_area_plot.figure.savefig(os.path.join(work_dir, 'negative_area.pdf'))

    # ___________________График_истинных_боксов____________________________
    gt_area_plot = df_gt_areas['gt_areas'].plot.hist(bins=200,
                                                     log=True,
                                                     figsize=(18, 9))
    gt_area_plot.set_title('Распределения площади истинных боксов', weight='bold', size=20)
    gt_area_plot.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    gt_area_plot.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    gt_area_plot.figure.savefig(os.path.join(work_dir, 'gt_area.pdf'))

    # ___________________График_проигнорированных_боксов_______________________
    gt_area_plot = df_unidentified.astype(float).plot.hist(bins=200,
                                                           log=True,
                                                           figsize=(18, 9),
                                                           xlim=(0, graph_scale))
    gt_area_plot.set_title('Распределения площади истинных боксов, которые были проигнорированы детектором',
                           weight='bold', size=20)
    gt_area_plot.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    gt_area_plot.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    gt_area_plot.figure.savefig(os.path.join(work_dir, 'unidentified_bbox_area.pdf'))

    # ___________________График_истинных_к_нераспознанным______________________
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot()

    ax.hist(df_gt_areas['gt_areas'], alpha=0.5, bins=200, color='red', label='ground_truth', log=True)
    ax.hist(df_unidentified['unidentified'], alpha=1, bins=200, label='unidentified', log=True)

    ax.legend(loc=1, fontsize=15)
    ax.set_title('Истинные и нераспознанные боксы', weight='bold', size=20)
    ax.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    ax.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    #ax.set_xlim(0,1000)
    plt.savefig(os.path.join(work_dir, 'Unidentified_to_ground_truth.pdf'))

    # ___________________График_истинных_к_неверно_распознанным_________________
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot()

    ax.hist(df_gt_areas['gt_areas'], alpha=0.5, bins=200, color='red', label='ground_truth', log=True)
    ax.hist(df_negative['negative'], alpha=1, bins=200, label='negative', log=True)

    ax.legend(loc=1, fontsize=15)
    ax.set_title('Истинные и неверно распознанные боксы', weight='bold', size=20)
    ax.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    ax.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    #ax.set_xlim(0,1000)
    plt.savefig(os.path.join(work_dir, 'Negative_to_ground_truth.pdf'))

    # ___________________График_истинных_к_верно_распознанным__________________
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot()

    ax.hist(df_gt_areas['gt_areas'], alpha=0.5, bins=200, color='red', label='ground_truth', log=True)
    ax.hist(df_positive['positive'], alpha=0.7, bins=200, label='positive', log=True)

    ax.legend(loc=1, fontsize=15)
    ax.set_title('Истинные и верно распознанные боксы', weight='bold', size=20)
    ax.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    ax.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    #ax.set_xlim(0,1000)
    plt.savefig(os.path.join(work_dir, 'Positive_to_ground_truth.pdf'))

    print('Сохраняем метрики')
    
    
    result_df = pd.DataFrame(index=['Total', 'TP', 'FP', 'FN', 'Precision', 'Recall'],
                         columns=['[0:10]', '(10:20]', '(20:30]', '(30:50]', '(50:100]', '(100:150]', '(150:200]', '(200:500]', '>500'])

    df_list = [df_gt_areas, df_positive, df_negative ,df_unidentified]   
    

    for i in range(len(df_list)):
        result_df.iloc[i][0] = df_list[i][df_list[i] <= 10].count()[0]
        result_df.iloc[i][1] = df_list[i][(df_list[i] > 10) & (df_list[i] <= 20)].count()[0]
        result_df.iloc[i][2] = df_list[i][(df_list[i] > 20) & (df_list[i] <= 30)].count()[0]
        result_df.iloc[i][3] = df_list[i][(df_list[i] > 30) & (df_list[i] <= 50)].count()[0]
        result_df.iloc[i][4] = df_list[i][(df_list[i] > 50) & (df_list[i] <= 100)].count()[0]
        result_df.iloc[i][5] = df_list[i][(df_list[i] > 100) & (df_list[i] <= 150)].count()[0]
        result_df.iloc[i][6] = df_list[i][(df_list[i] > 150) & (df_list[i] <= 200)].count()[0]
        result_df.iloc[i][7] = df_list[i][(df_list[i] > 200) & (df_list[i] <= 500)].count()[0]
        result_df.iloc[i][8] = df_list[i][df_list[i] > 500].count()[0]
    for j in range(len(result_df.columns)):
        result_df.iloc[4][j] = round(result_df.iloc[1][j] / (result_df.iloc[1][j] + result_df.iloc[2][j]), 3)
        result_df.iloc[5][j] = round(result_df.iloc[1][j] / (result_df.iloc[1][j] + result_df.iloc[3][j]), 3)

    
    result_df.to_csv(os.path.join(work_dir, 'result_metrics.csv'))

    print('Готово')
    
def parse_args():
    parser = argparse.ArgumentParser(description='Сбор статистики площади отрисованных боксам')
    parser.add_argument('result_pkl', help='Файл с результатами, созданный tools/test.py')
    parser.add_argument('annotations', help='Файл аннотаций JSON')
    parser.add_argument('out', default='stats.json', help='Название файла, куда сохраняем результаты')
    parser.add_argument('workdir', help='Папка, куда сохраняем графики')
    parser.add_argument('--iou_thr', default=0.3, help='Порог, после которого бокс идет на отрисовку')
    parser.add_argument('--ignore_thr', default=0.03, help='Минимальный порог для того чтобы истинный бокс'
                                                           'не отправилься в неопознанные')
    parser.add_argument('--graph_scale', default=1000, help='xlim у графиков')
    args = parser.parse_args()
    return args


def main():
    # Парсим аргументы
    args = parse_args()
    # Считаем статистики
    statistic_2(
        result_pkl=args.result_pkl,
        annotation_json=args.annotations,
        out=args.out,
        iou_thr=args.iou_thr
    )
    # Сохраняем графики
    plot_graphs(
        statistic_json=args.out,
        iou_thr=args.iou_thr,
        ignore_thr=args.ignore_thr,
        graph_scale=args.graph_scale,
        work_dir=args.workdir
    )


if __name__ == '__main__':
    main()
