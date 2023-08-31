import os
import xml.etree.ElementTree as ET
from shapely import Polygon

def txt2dict(txt_path: str, split_word: str = '|'):
    """ txt 파일을 딕셔너리로 파싱
    
    """
    result = {}
    file = open(txt_path, 'r')
    for line in file:
        words = line.split(split_word)
        num = words[0]
        cls = words[1].replace('\n', '')
        result[num] = cls
    return result

def xml2dict(element):
    """ xml 파일을 딕셔너리로 파싱
    
    """
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
    """ xml 파일들을 딕셔너리로 파싱
    
    """
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
    """ IoU 계산 (바운딩 박스가 회전되어 있으므로 shapely.Polygon 사용)

    """
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
    """ Precision, Recall 계산에 필요한 TP, DT, GT 카운팅
    
    Arguments:
        gt_dict: GT xml로부터 파싱된 딕셔너리
        dt_dict: DT xml로부터 파싱된 딕셔너리
        iou_thr: IoU Threshold

    Returns:
        precision: {도면 이름: {클래스 이름: {tp, dt}}, ..., total: {tp, dt}}}}
        recall: {도면 이름: {클래스 이름: {tp, gt}}, ..., total: {tp, gt}}}} 
    
    """
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

def dump(dump_path: str, gt_xmls_path: str, symbol_dict: dict, precision: dict, recall: dict, recognition: dict = {}, symbol: str = 'total'):
    """ Precision, Recall을 계산하여 txt 파일로 출력
    
    Arguments:
        symbol(text)은 'total', 'small' 또는 'large' 중 해당되는 symbol에 대해 dump 수행

        dump_path: result.txt를 저장할 경로
        gt_xmls_path: GT 도면 xml의 경로 (도면 이름을 얻기 위함)

        precision: {도면 이름: {클래스 이름: {tp, dt}}, ..., total: {tp, dt}}}}
        recall: {도면 이름: {클래스 이름: {tp, gt}}, ..., total: {tp, gt}}}}
        recognition: {도면 이름: {tp, dt}, ..., total: {tp, gt}}?
    
    """
    mean = {}
    mean['tp'] = 0
    mean['dt'] = 0
    mean['gt'] = 0

    # result_file = open(f"{dump_path}\\result.txt", 'a')
    result_file = open(f"{dump_path}/result.txt", 'a')
    result_file.write(f"Symbol Type: {symbol}\n")
    
    for root, dirs, files in os.walk(gt_xmls_path):
        for filename in files:
            if filename.endswith('.xml'):
                diagram = filename[0:22]
                result_file.write(f"\n")

                result_file.write(f'test drawing: {diagram}----------------------------------\n')

                tp = precision[diagram]['total']['tp']
                dt = precision[diagram]['total']['dt']
                pr = tp / dt if dt != 0 else 0
                result_file.write(f"total precision: {tp} / {dt} = {pr}\n")

                tp = recall[diagram]['total']['tp']
                gt = recall[diagram]['total']['gt']
                rc = tp / gt if gt != 0 else 0
                result_file.write(f"total recall : {tp} / {gt} = {rc}\n")

                mean['tp'] += tp
                mean['dt'] += dt
                mean['gt'] += gt

                for key, value in symbol_dict.items():
                    if value in recall[diagram]:
                        tp = recall[diagram][value]['tp']
                        gt = recall[diagram][value]['gt']
                        result_file.write(f"class {key} (['{value}']): {tp} / {gt}\n")
                
                result_file.write(f'\n')

    mean['precision'] = mean['tp'] / mean['dt'] if mean['dt'] != 0 else 0
    mean['recall'] = mean['tp'] / mean['gt'] if mean['gt'] != 0 else 0
    result_file.write(f"(mean precision, mean recall) = ({mean['precision']}, {mean['recall']})")

    result_file.close()
    return

# pipeline

gt_xmls = 'D:\\Data\\xml2eval\\GT_xmls\\'
dt_xmls = 'D:\\Data\\xml2eval\\DT_xmls\\'
symbol_txt = 'D:\\Data\\SymbolClass_Class.txt'
dump_path = 'D:\\Experiments\\Detections\\'

gt_dict = xmls2dict(gt_xmls)
dt_dict = xmls2dict(dt_xmls)
symbol_dict = txt2dict(symbol_txt)

precision, recall = evaluate(gt_dict, dt_dict)

dump(dump_path, gt_xmls, symbol_dict, precision, recall) # symbol 종류에 따라 해당되는 symbol에 대해서만 dump 수행 가능해야 함