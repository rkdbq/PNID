# 각 도면(xml)의 precision, recall 계산
# 전체 도면(xml)에 대한 precision, recall 계산
# 계산 시 모든 심볼/작은 심볼/큰 심볼 선택 가능하여야 함
# 계산 결과를 텍스트로 출력 

import os
import xml.etree.ElementTree as ET
from shapely import Polygon

def xml2dict(element):
    result = {}
    for child in element:
        child_data = xml2dict(child)
        if child_data:
            if child.tag in result:
                if type(result[child.tag]) is list:
                    result[child.tag].append(child_data)
                else:
                    result[child.tag] = [result[child.tag], child_data]
            else:
                result[child.tag] = child_data
        else:
            result[child.tag] = child.text
    return result

def xmls2dict(xml_dir_path: str):
    result = {}
    for root, dirs, files in os.walk(xml_dir_path):
        for filename in files:
            if filename.endswith('.xml'):
                file_path = os.path.join(root, filename)
                try:
                    tree = ET.parse(file_path)
                    root_element = tree.getroot()
                    result[filename[0:22]] = xml2dict(root_element)['symbol_object']
                except ET.ParseError as e:
                    print(f'Error parsing {file_path}: {e}')
    return result

def cal_iou(gt_points: dict, dt_points: dict):
    xmin = float(gt_points['xmin'])
    xmax = float(gt_points['xmax'])
    ymin = float(gt_points['ymin'])
    ymax = float(gt_points['ymax'])
    coords = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),]
    gt_rect = Polygon(coords)

    xmin = dt_points['xmin']
    xmax = dt_points['xmax']
    ymin = dt_points['ymin']
    ymax = dt_points['ymax']
    coords = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),]
    dt_rect = Polygon(coords)

    intersection = gt_rect.intersection(dt_rect).area
    union = gt_rect.union(dt_rect).area
    iou = intersection / union
    return iou

def evaluate(gt_dict: dict, dt_dict: dict, iou_thr: float = 0.8):
    precision = {}
    recall = {}

    for diagram in gt_dict.keys():
        precision[diagram] = {}
        recall[diagram] = {}
        precision[diagram]['total'] = {}
        recall[diagram]['total'] = {}
        precision[diagram]['total']['tp'] = 0
        precision[diagram]['total']['dt'] = 0
        recall[diagram]['total']['tp'] = 0
        recall[diagram]['total']['gt'] = 0
        if diagram not in dt_dict: 
            print(f'{diagram} is skipped. (NOT exist in detection xmls path)\n')
            continue
        
        # Counting tp, dt, gt for each classes
        tp = {}
        dt = {}
        gt = {}
        for gt_bbox in gt_dict[diagram]:
            for dt_bbox in dt_dict[diagram]:
                if gt_bbox['class'] == dt_bbox['class']:
                    cls = gt_bbox['class']
                    if cls not in tp:
                        tp[cls] = 0
                    iou = cal_iou(gt_bbox['bndbox'], dt_bbox['bndbox'])
                    if iou > iou_thr:
                        tp[cls] += 1
        for dt_bbox in dt_dict[diagram]:
            cls = dt_bbox['class']
            if cls not in dt:
                dt[cls] = 0
            dt[cls] += 1
        for gt_bbox in gt_dict[diagram]:
            cls = gt_bbox['class']
            if cls not in gt:
                gt[cls] = 0
            gt[cls] += 1

        # Mapping precision
        for cls, cnt in tp.items():
            if cls not in precision[diagram]:
                precision[diagram][cls] = {}
            precision[diagram][cls]['tp'] = cnt
            precision[diagram]['total']['tp'] += cnt
        for cls, cnt in dt.items():
            if cls not in precision[diagram]:
                precision[diagram][cls] = {}
            precision[diagram][cls]['dt'] = cnt
            precision[diagram]['total']['dt'] += cnt

        # Mapping recall
        for cls, cnt in tp.items():
            if cls not in recall[diagram]:
                recall[diagram][cls] = {}
            recall[diagram][cls]['tp'] = cnt   
            recall[diagram]['total']['tp'] += cnt
        for cls, cnt in gt.items():
            if cls not in recall[diagram]:
                recall[diagram][cls] = {}
            recall[diagram][cls]['gt'] = cnt
            recall[diagram]['total']['gt'] += cnt
            
    return precision, recall

def dump(path: str, gt_xmls_path: str, precision: dict, recall: dict, recognition: dict = {}, symbol: str = 'total'):
    """
    symbol(text) is 'total', 'small' or 'large'

    precision(dict): {도면 이름: {클래스 이름: {tp, dt}}, ... total: {tp, dt}}}, ..., mean: sum(tp) / sum(dt)}
    recall(dict): {도면 이름: {클래스 이름: {tp, gt}}, ... total: {tp, gt}}}, ..., mean: sum(tp) / sum(gt)}
    recognition(dict): {도면 이름: {tp, dt}, ..., mean: {sum(tp), sum(dt)}}
    """

    result_file = open(f'{path}\\result.txt', 'a')
    result_file.write(f'Symbol Type: {symbol}\n')
    
    for root, dirs, files in os.walk(gt_xmls_path):
        for filename in files:
            if filename.endswith('.xml'):
                diagram = filename[0:22]
                result_file.write(f'\n')
                result_file.write(f'test drawing: {diagram}----------------------------------\n')
                tp = precision[diagram]['total']['tp']
                dt = precision[diagram]['total']['dt']
                result_file.write(f"total precision: {tp}/{dt} = {tp / dt if dt != 0 else 0}\n")
                
                result_file.write(f'\n')

    result_file.close()
    return

# pipeline

gt_xmls = 'D:\\Data\\xml2eval\\GT_xmls\\'
dt_xmls = 'D:\\Data\\xml2eval\\DT_xmls\\'
dump_path = 'D:\\Experiments\\Detections\\'

gt_dict = xmls2dict(gt_xmls)
dt_dict = xmls2dict(dt_xmls)

precision, recall = evaluate(gt_dict, dt_dict)

dump(dump_path, gt_xmls, precision, recall)