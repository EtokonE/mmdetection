import pickle
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
        The (x1, y1) координаты левого верхнего угла,
        the (x2, y2) координаты правого нижнего угла
    bb2 : list
        [x1, y1, x2, y2]
        The (x, y) координаты левого верхнего угла,
        the (x2, y2) координаты правого нижнего угла

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

    # Вычисляем координаты прямоугольника пересечения
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Вычисляем пересечение двух боксов
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Вычисляем объединение боксов
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # Рассчитываем iou
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def read_inference_file(results_pkl):
    '''Загружаем файл, созданный при помощи tools/test.py'''
    with open(str(results_pkl), 'rb') as pkl:
        data_pkl = pickle.load(pkl)
    return data_pkl


def read_annotation_file(annotation_json):
    '''Загружаем файл аннотаций'''
    with open(str(annotation_json)) as js:
        data_js = json.load(js)
    return data_js


def calculate_statistics(result_pkl, annotation_json, out, confidence, work_dir):
    '''Вычисляем необходимые статистики и записываем в файл'''

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # Загружаем файл, созданный при помощи tools/test.py
    data_pkl = read_inference_file(result_pkl)

    # Загружаем файл аннотаций, будем добавлять информацию о статистиках в эту переменную
    data_js = read_annotation_file(annotation_json)
    assert len(data_pkl) == len(data_js)

    for i in range(len(data_js)):
        # Создаем переменные для хранения координат, площади сгенерированных боксов, и IOU боксов прошедших через порог confidence
        predicted_bbox = {}
        predicted_bbox['bboxes'] = []
        predicted_bbox['areas'] = []
        predicted_bbox['confidence'] = []

        # Вычисляем площадь ground_truth боксов
        data_js[i]['area'] = []
        for num_bbox in range(len(data_js[i]['ann']['bboxes'])):
            data_js[i]['area'].append(abs(
                        (data_js[i]['ann']['bboxes'][num_bbox][0] - data_js[i]['ann']['bboxes'][num_bbox][2]) * (
                        data_js[i]['ann']['bboxes'][num_bbox][1] - data_js[i]['ann']['bboxes'][num_bbox][3])))

        # Вычисляем статистику по сгенерированным боксам
        for j in range(len(data_pkl[i][0])):
            # Проверяем порог истинности бокса
            if data_pkl[i][0][j][4] < confidence:
                break
            elif data_pkl[i][0][j][4] >= confidence:
                # Добавляем координаты сгенерированных боксов
                predicted_bbox['bboxes'].append(data_pkl[i][0][j][:-1].tolist())
                # Добавляем площадь сгенерированных боксов
                predicted_bbox['areas'].append(str(abs(
                    (data_pkl[i][0][j][0] - data_pkl[i][0][j][2]) * (data_pkl[i][0][j][1] - data_pkl[i][0][j][3]))))
                # Добавляем сonfidence сгенерированных боксов
                predicted_bbox['confidence'].append(str(round(data_pkl[i][0][j][4], 2)))
        data_js[i]['generated_bbox'] = predicted_bbox

        # Определяем истинность боксов
        bb1 = data_js[i]['ann']['bboxes']
        bb2 = data_js[i]['generated_bbox']['bboxes']

        # Вычисляем максимальный IOU предсказанного бокса с одним из истинных
        ious = []
        for pred in bb2:
            temp = 0
            for gt in bb1:
                calculate_iou = get_iou(gt, pred)
                if calculate_iou >= temp:
                    temp = calculate_iou
            ious.append(temp)
        data_js[i]['generated_bbox']['iou_predicted_to_gt'] = ious

        # Вычисляем максимальный IOU истинного бокса с одним из предсказанных
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



def make_statistic_table(statistic_json, iou_thr):
    '''
    Распределяем данные о площадях боксов в зависимости от положения в Consolution Matrix
    Parameters
    ----------
    statistic_json : json file
    iou_thr : float
        Порог IOU для определения истинности бокса
    Returns
    -------
    df_gt_areas : DataFrame
        Площадь всех Ground Truth боксов
    positive_bbox_area : DataFrame
        Площадь всех True Positive
    negative_bbox_area : DataFrame
        Площадь всех False Positive
    unidentified : DataFrame
        Площадь всех False Negative
    '''
    with open(str(statistic_json)) as st:
        stats = json.load(st)
        # Площадь всех ground_truth боксов
        gt_areas = [image['area'][index_bbox] for image in stats for index_bbox in range(len(image['area']))]
        df_gt_areas = pd.DataFrame({'gt_areas': gt_areas})

        # Площадь отрисованных боксов
        annotations = [image for image in stats]
        positive_bbox_area = []
        negative_bbox_area = []
        unidentified = []
        for anno in annotations:
            for i in range(len(anno.get('generated_bbox').get('iou_predicted_to_gt'))):
                # Площадь неправильно задетектированных боксов (FP)
                if float(anno.get('generated_bbox').get('iou_predicted_to_gt')[i]) < iou_thr:
                    negative_bbox_area.append(float(anno.get('generated_bbox').get('areas')[i]))

            for i in range(len(anno.get('generated_bbox').get('iou_gt_to_predicted'))):
                # Площадь правильно задетектированных боксов (TP)
                if float(anno.get('generated_bbox').get('iou_gt_to_predicted')[i]) >= iou_thr:
                    positive_bbox_area.append(float(anno.get('area')[i]))

                # Площадь истинных боксов, которые быди проигнорированны детектором (FN)
                elif float(anno.get('generated_bbox').get('iou_gt_to_predicted')[i]) < iou_thr:
                    unidentified.append(float(anno.get('area')[i]))

        df_positive = pd.DataFrame({'positive': positive_bbox_area})
        df_negative = pd.DataFrame({'negative': negative_bbox_area})
        df_unidentified = pd.DataFrame({'unidentified': unidentified})
    return df_gt_areas, df_positive, df_negative, df_unidentified


def plot_single_graph(bbox_area_df, graph_scale, bbox_type, work_dir):
    '''
    Сохраняем график распределение площади боксов

    Parameters
    ----------
    bbox_area_df : Data Frame
        Датафрейм, сгенерированный в make_statistic_table()
    graph_scale : int
        Верхнее ограничение графика по оси абсцисс
    bbox_type : str
        Вставка в заголовок графика для описания типа боксов
    work_dir : str
        Директория для сохранения графика
    '''
    gt_area_plot = bbox_area_df.plot.hist(bins=200,
                                         log=True,
                                         figsize=(18, 9),
                                         xlim=(0, graph_scale)
                                         )
    gt_area_plot.set_title(f'Распределения площади {bbox_type} боксов', weight='bold', size=20)
    gt_area_plot.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    gt_area_plot.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    gt_area_plot.figure.savefig(os.path.join(work_dir, f'{bbox_type}.png'))

def plot_relative_graph(bbox_area_df_first, bbox_area_df_second, title, graph_scale, work_dir):
    '''
    Сохраняем график с наложенными друг на друга распределениями площади бовсов двух разных типов

    Parameters
    ----------
    bbox_area_df_first : Data Frame
        Первый датафрейм, сгенерированный в make_statistic_table()
    bbox_area_df_second : Data Frame
        Второй датафрейм, сгенерированный в make_statistic_table()
    title : str
        Заголовок графика
    graph_scale : int
        Верхнее ограничение графика по оси абсцисс
    work_dir : str
        Директория для сохранения графика
    '''
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot()

    ax.hist(bbox_area_df_first.iloc[:,0], alpha=0.5, bins=200, color='red', label=f'{bbox_area_df_first.columns[0]}', log=True)
    ax.hist(bbox_area_df_second.iloc[:,0], alpha=1, bins=200, label=f'{bbox_area_df_second.columns[0]}', log=True)

    ax.legend(loc=1, fontsize=15)
    ax.set_title(str(title), weight='bold', size=20)
    ax.set_xlabel('Площадь бокса', labelpad=20, weight='bold', size=12)
    ax.set_ylabel('Количество (log)', labelpad=20, weight='bold', size=12)
    ax.set_xlim(0, graph_scale)
    plt.savefig(os.path.join(work_dir, f'{str(title)}.png'))


def make_table_metric(df_gt_areas, df_positive, df_negative, df_unidentified, work_dir, iou_thr):
    '''Сохраняем сводные статистики и метрики Precision, Recall в таблице'''
    result_df = pd.DataFrame(index=['Total', 'TP', 'FP', 'FN', 'Precision', 'Recall'],
                             columns=['[0:10]', '(10:20]', '(20:30]', '(30:50]', '(50:100]', '(100:150]', '(150:200]',
                                      '(200:500]', '>500'])

    df_list = [df_gt_areas, df_positive, df_negative, df_unidentified]

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

    result_df.to_csv(os.path.join(work_dir, f'result_metrics_{str(iou_thr)}.csv'))


def parse_args():
    parser = argparse.ArgumentParser(description='Сбор статистики площади отрисованных боксам')
    parser.add_argument('result_pkl', help='Файл с результатами, созданный tools/test.py')
    parser.add_argument('annotations', help='Файл аннотаций JSON')
    parser.add_argument('out', default='stats.json', help='Название файла, куда сохраняем результаты')
    parser.add_argument('work_dir', help='Папка, куда сохраняем графики')
    parser.add_argument('--confidence', default=0.3, help='Порог, после которого бокс идет на отрисовку')
    parser.add_argument('--iou_thr', default=0.3, help='Минимальный порог iou')
    parser.add_argument('--graph_scale', default=1000, help='xlim у графиков')
    args = parser.parse_args()
    return args


def main():
    # Парсим аргументы
    args = parse_args()
    # Сохраняем статистики в json файл
    calculate_statistics(args.result_pkl, args.annotations, args.out, args.confidence, args.work_dir)
    # Разбиваем данные на датафреймы в зависимости от положения в Conv Matrix
    df_gt_areas, df_positive, df_negative, df_unidentified = make_statistic_table(args.out, args.iou_thr)
    # Сохраняем одиночные графики
    plot_single_graph(bbox_area_df=df_gt_areas, graph_scale=args.graph_scale, bbox_type='GT', work_dir=args.work_dir)
    plot_single_graph(bbox_area_df=df_positive, graph_scale=args.graph_scale, bbox_type='TP', work_dir=args.work_dir)
    plot_single_graph(bbox_area_df=df_negative, graph_scale=args.graph_scale, bbox_type='FP', work_dir=args.work_dir)
    plot_single_graph(bbox_area_df=df_unidentified, graph_scale=args.graph_scale, bbox_type='FN', work_dir=args.work_dir)
    # Сохраняем наложенные графики
    plot_relative_graph(bbox_area_df_first=df_gt_areas, bbox_area_df_second=df_unidentified,
                        title='GT_to_FN', graph_scale=args.graph_scale, work_dir=args.work_dir)
    plot_relative_graph(bbox_area_df_first=df_gt_areas, bbox_area_df_second=df_negative,
                        title='GT_to_FP', graph_scale=args.graph_scale, work_dir=args.work_dir)
    plot_relative_graph(bbox_area_df_first=df_gt_areas, bbox_area_df_second=df_positive,
                        title='GT_to_TP', graph_scale=args.graph_scale, work_dir=args.work_dir)
    # Сохраняем таблицу с метриками
    make_table_metric(df_gt_areas, df_positive, df_negative, df_unidentified, work_dir=args.work_dir, iou_thr=args.iou_thr)


if __name__ == '__main__':
    main()