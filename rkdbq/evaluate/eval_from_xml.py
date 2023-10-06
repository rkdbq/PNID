import os, cv2, math
import xml.etree.ElementTree as ET
import numpy as np
from shapely import Polygon
from pathlib import Path
from tqdm import tqdm

class evaluate_from_xml():
    def __init__(self, gt_xmls_path: str, dt_xmls_path: str, symbol_txt_path: str, large_symbol_txt_path: str, iou_thr: float = 0.8):
        self.__xmls_path = {}
        self.__xmls_path['gt'] = gt_xmls_path
        self.__xmls_path['dt'] = dt_xmls_path
        self.__symbol_txt_path = {}
        self.__symbol_txt_path['total'] = symbol_txt_path
        self.__symbol_txt_path['large'] = large_symbol_txt_path
        self.__iou_thr = iou_thr

    def __diff_dict(self, remain: dict, remove: dict):
        """ remain 딕셔너리 중 remove 딕셔너리와 중복되는 키를 가지는 쌍을 삭제
        
        """
        diff = {}
        for key in remain.keys():
            if key not in remove:
                diff[key] = remain[key]
        return diff

    def __txt2dict(self, txt_path: str, split_word: str = '|'):
        """ txt 파일을 딕셔너리로 파싱
        
        """
        result = {}
        file = open(txt_path, 'r')
        for line in file:
            words = line.split(split_word)
            num = words[0]
            cls = words[1].replace('\n', '')
            result[cls] = num
        return result
    
    def __dict2points(self, points: dict):
        """ 점 딕셔너리를 리스트로 변환
        
        """
        coords = [round(float(points['x1'])), round(float(points['y1'])), round(float(points['x2'])), round(float(points['y2'])), 
                  round(float(points['x3'])), round(float(points['y3'])), round(float(points['x4'])), round(float(points['y4'])),]
        coords = np.array([int(i) for i in coords])
        coords = coords.reshape(4,2)
        coords = coords.tolist()
        return coords
    
    def __rotate(self, point: tuple, degree: float, pivot: tuple):
        """ pivot을 기준으로 degree만큼 point를 회전
        
        """
        rad = math.radians(degree)
        cos_theta = math.cos(rad)
        sin_theta = math.sin(rad)
        x, y = point
        piv_x, piv_y = pivot

        rotated = {}
        rotated['x'] = cos_theta * (x - piv_x) - sin_theta * (y - piv_y) + piv_x
        rotated['y'] = sin_theta * (x - piv_x) + cos_theta * (y - piv_y) + piv_y

        return (rotated['x'], rotated['y'])
    
    def __two2four(self, bbox: dict):
        """ 2점 좌표 형식을 4점 좌표 형식으로 변환
        
        """
        result = {}
        bbox_two = {}
        bbox_four = {}

        for key, value in bbox.items():
            result[key] = value
        for coord, value in bbox['bndbox'].items():
            bbox_two[coord] = float(value)

        mid = {}
        mid['x'] = (bbox_two['xmin'] + bbox_two['xmax']) / 2
        mid['y'] = (bbox_two['ymin'] + bbox_two['ymax']) / 2
        
        point = (bbox_two['xmin'], bbox_two['ymin'])
        degree = float(bbox['degree'])
        pivot = (mid['x'], mid['y'])
        x, y = self.__rotate(point, degree, pivot)
        bbox_four['x1'] = str(x)
        bbox_four['y1'] = str(y)   

        point = (bbox_two['xmin'], bbox_two['ymax'])
        x, y = self.__rotate(point, degree, pivot)
        bbox_four['x2'] = str(x)
        bbox_four['y2'] = str(y)

        point = (bbox_two['xmax'], bbox_two['ymax'])
        x, y = self.__rotate(point, degree, pivot)
        bbox_four['x3'] = str(x)
        bbox_four['y3'] = str(y)

        point = (bbox_two['xmax'], bbox_two['ymin'])
        x, y = self.__rotate(point, degree, pivot)
        bbox_four['x4'] = str(x)
        bbox_four['y4'] = str(y)

        result['bndbox'] = bbox_four

        return result
    
    def __xml2dict(self, element):
        """ xml 파일을 딕셔너리로 파싱
        
        """
        result = {}
        for child in element:
            child_data = self.__xml2dict(child)
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

    def __xmls2dict(self, xml_dir_path: str, mode: str):
        """ xml 파일들을 딕셔너리로 파싱
        
        """
        result = {}
        for root, dirs, files in os.walk(xml_dir_path):
            for filename in tqdm(files, f"Parcing XMLs ({mode})"):
                if filename.endswith('.xml'):
                    file_path = os.path.join(root, filename)
                    try:
                        tree = ET.parse(file_path)
                        root_element = tree.getroot()
                        diagram = filename[0:22]
                        result[diagram] = self.__xml2dict(root_element)['symbol_object']
                        if mode == 'dt':
                            four_bboxs = []
                            for bbox in result[diagram]:
                                four_bboxs.append(self.__two2four(bbox))
                            result[diagram] = four_bboxs
                    except ET.ParseError as e:
                        print(f'Error parsing {file_path}: {e}')
        return result

    def __cal_iou(self, gt_points: dict, dt_points: dict):
        """ IoU 계산 (바운딩 박스가 회전되어 있으므로 shapely.Polygon 사용)

        """
        coords = self.__dict2points(gt_points)
        gt_rect = Polygon(coords)

        coords = self.__dict2points(dt_points)
        dt_rect = Polygon(coords)

        iou = 0
        if gt_rect.intersects(dt_rect):
            intersection = gt_rect.intersection(dt_rect).area
            union = gt_rect.union(dt_rect).area
            iou = intersection / union
        return iou

    def __evaluate(self, gt_dict: dict, dt_dict: dict, symbol_dict: dict,  cmp_degree: bool = False, cmp_recog: bool = False):
        """ Precision, Recall 계산에 필요한 TP, DT, GT 카운팅
        
        Arguments:
            gt_dict: GT xml로부터 파싱된 딕셔너리
            dt_dict: DT xml로부터 파싱된 딕셔너리
            symbol_dict: Symbol txt로부터 파싱된 딕셔너리
            cmp_degree: 각도 비교 여부

        Returns:
            precision: {도면 이름: {클래스 이름: {tp, dt}}, ..., total: {tp, dt}}}}
            recall: {도면 이름: {클래스 이름: {tp, gt}}, ..., total: {tp, gt}}}} 
        
        """
        precision = {}
        recall = {}
        degree = {}
        recognition = {}
        for diagram in tqdm(gt_dict.keys(), f"Evaluation"):
            precision[diagram] = {}
            precision[diagram]['total'] = {}
            precision[diagram]['total']['tp'] = 0
            precision[diagram]['total']['dt'] = 0
            recall[diagram] = {}
            recall[diagram]['total'] = {}
            recall[diagram]['total']['tp'] = 0
            recall[diagram]['total']['gt'] = 0
            degree[diagram] = {}
            degree[diagram]['total'] = {}
            degree[diagram]['total']['tp_with_deg'] = 0
            degree[diagram]['total']['tp'] = 0
            # recognition[diagram] = {}
            # recognition[diagram]['total'] = {}
            # recognition[diagram]['total']['tp_with_recog'] = 0
            # recognition[diagram]['total']['tp'] = 0
            if diagram not in dt_dict: 
                print(f'{diagram} is skipped. (NOT exist in detection xmls path)\n')
                continue
            
            # Counting tp, dt, gt for each classes
            tp = {}
            tp_with_deg = {}
            # tp_with_recog = {}
            dt = {}
            gt = {}
            for gt_item in gt_dict[diagram]:
                for dt_item in dt_dict[diagram]:
                    if gt_item['type'] == dt_item['type']:
                        if gt_item['type'] == 'text' or gt_item['class'] == dt_item['class']:
                            cls = gt_item['class'] if gt_item['type'] != 'text' else 'text'
                            if cls not in symbol_dict:
                                continue
                            if cls not in tp:
                                tp[cls] = 0
                            if cls not in tp_with_deg:
                                tp_with_deg[cls] = 0
                            # if cls not in tp_with_recog:
                            #     tp_with_recog[cls] = 0
                            iou = self.__cal_iou(gt_item['bndbox'], dt_item['bndbox'])
                            if iou > self.__iou_thr:
                                tp[cls] += 1
                                # print(f"gt: {gt_item['degree']}, dt: {dt_item['degree']}")
                                gt_deg = float(gt_item['degree'])
                                dt_deg = float(dt_item['degree'])
                                if cmp_degree and gt_deg == dt_deg:
                                    tp_with_deg[cls] += 1
                                # if cmp_recog and gt_bbox['type'] gt_recog == dt_recog:
                                #     tp_with_recog[cls] += 1
            for dt_item in dt_dict[diagram]:
                cls = dt_item['class'] if dt_item['type'] != 'text' else 'text'
                if cls not in symbol_dict:
                    continue
                if cls not in dt:
                    dt[cls] = 0
                dt[cls] += 1
            for gt_item in gt_dict[diagram]:
                cls = gt_item['class'] if gt_item['type'] != 'text' else 'text'
                if cls not in symbol_dict:
                    continue
                if cls not in gt:
                    gt[cls] = 0
                gt[cls] += 1

            # Mapping precision
            for cls, cnt in tp.items():
                if cls not in symbol_dict: 
                    continue
                if cls not in precision[diagram]:
                    precision[diagram][cls] = {}
                precision[diagram][cls]['tp'] = cnt
                precision[diagram]['total']['tp'] += cnt
            for cls, cnt in dt.items():
                if cls not in symbol_dict:
                    continue
                if cls not in precision[diagram]:
                    precision[diagram][cls] = {}
                precision[diagram][cls]['dt'] = cnt
                precision[diagram]['total']['dt'] += cnt
                if 'tp' not in precision[diagram][cls]: 
                    precision[diagram][cls]['tp'] = 0

            # Mapping recall
            for cls, cnt in tp.items():
                if cls not in symbol_dict:
                    continue
                if cls not in recall[diagram]:
                    recall[diagram][cls] = {}
                recall[diagram][cls]['tp'] = cnt   
                recall[diagram]['total']['tp'] += cnt
            for cls, cnt in gt.items():
                if cls not in symbol_dict:
                    continue
                if cls not in recall[diagram]:
                    recall[diagram][cls] = {}
                recall[diagram][cls]['gt'] = cnt
                recall[diagram]['total']['gt'] += cnt
                if 'tp' not in recall[diagram][cls]: 
                    recall[diagram][cls]['tp'] = 0

            # Mapping degree
            if cmp_degree:
                for cls, cnt in tp_with_deg.items():
                    if cls not in symbol_dict:
                        continue
                    if cls not in degree[diagram]:
                        degree[diagram][cls] = {}
                    degree[diagram][cls]['tp_with_deg'] = cnt   
                    degree[diagram]['total']['tp_with_deg'] += cnt
                for cls, cnt in tp.items():
                    if cls not in symbol_dict:
                        continue
                    if cls not in degree[diagram]:
                        degree[diagram][cls] = {}
                    degree[diagram][cls]['tp'] = cnt
                    degree[diagram]['total']['tp'] += cnt
                    if 'tp_with_deg' not in degree[diagram][cls]: 
                        degree[diagram][cls]['tp_with_deg'] = 0
                
        return precision, recall, degree, recognition

    def dump(self, dump_path: str, symbol_type: str = 'total',  cmp_degree: bool = False, cmp_recog: bool = False):
        """ Precision, Recall을 계산하여 txt 파일로 출력
        
        Arguments:
            symbol_type(text)은 'total', 'small' 또는 'large'
            dump_path: result.txt를 저장할 경로
            cmp_degree: 각도 비교 여부
        
        """

        gt_dict = self.__xmls2dict(self.__xmls_path['gt'], mode='gt')
        dt_dict = self.__xmls2dict(self.__xmls_path['dt'], mode='dt')

        symbol_dict = {}
        symbol_dict['total'] = self.__txt2dict(self.__symbol_txt_path['total'])
        symbol_dict['large'] = self.__txt2dict(self.__symbol_txt_path['large'])
        symbol_dict['small'] = self.__diff_dict(symbol_dict['total'], symbol_dict['large'])

        symbol_dict = symbol_dict[symbol_type]
        precision, recall, degree, recognition = self.__evaluate(gt_dict, dt_dict, symbol_dict, cmp_degree, cmp_recog)

        mean = {}
        mean['tp'] = 0
        mean['dt'] = 0
        mean['gt'] = 0
        mean['tp_with_deg'] = 0

        Path(dump_path).mkdir(parents=True, exist_ok=True)
        result_file = open(f"{dump_path}\\result_{symbol_type}.txt", 'w')
        result_file.write(f"Symbol Type: {symbol_type}\n")
        result_file.write(f"IoU Threshold: {self.__iou_thr}\n")
        
        for diagram in tqdm(gt_dict.keys(), "Writing"):
            result_file.write(f"\n")
            result_file.write(f'test drawing: {diagram}----------------------------------\n')

            score = "precision"
            tp = precision[diagram]['total']['tp']
            dt = precision[diagram]['total']['dt']
            pr = tp / dt if dt != 0 else 0
            result_file.write(f"total {score}: {tp} / {dt} = {pr}\n")

            score = "recall"
            tp = recall[diagram]['total']['tp']
            gt = recall[diagram]['total']['gt']
            rc = tp / gt if gt != 0 else 0
            result_file.write(f"total {score}: {tp} / {gt} = {rc}\n")

            mean['tp'] += tp
            mean['dt'] += dt
            mean['gt'] += gt

            if cmp_degree:
                score = "degree coreection ratio"
                tp_with_deg = degree[diagram]['total']['tp_with_deg']
                tp = degree[diagram]['total']['tp']
                dg_ratio = tp_with_deg / tp if tp != 0 else 0
                result_file.write(f"total {score}: {tp_with_deg} / {tp} = {dg_ratio}\n")
            
                mean['tp_with_deg'] += tp_with_deg

            for cls, num in symbol_dict.items():
                if cls in recall[diagram]:
                    tp = recall[diagram][cls]['tp']
                    gt = recall[diagram][cls]['gt']
                    if cmp_degree and cls in degree[diagram]:
                        tp_with_deg = degree[diagram][cls]['tp_with_deg']
                        tp = degree[diagram][cls]['tp']
                        result_file.write(f"class {num} (['{cls}']): {tp} / {gt}, {score}: {tp_with_deg} / {tp}\n")
                    else: 
                        result_file.write(f"class {num} (['{cls}']): {tp} / {gt}\n")            
            result_file.write('\n')

        mean['precision'] = mean['tp'] / mean['dt'] if mean['dt'] != 0 else 0
        mean['recall'] = mean['tp'] / mean['gt'] if mean['gt'] != 0 else 0
        result_file.write(f"(mean precision, mean recall) = ({mean['precision']}, {mean['recall']})\n")
        
        if cmp_degree:
            mean['degree_correction_ratio'] = mean['tp_with_deg'] / mean['tp'] if mean['tp'] != 0 else 0
            result_file.write(f"(mean degree correction ratio) = ({mean['degree_correction_ratio']})\n")

        result_file.close()
        return
    
    def visualize(self, gt_imgs_path: str, write_path: str, cls: str = 'text'):
        """ 바운딩 박스들을 가시화
        
        Arguments:
            gt_imgs_path: 원본 이미지의 경로
            write_path: 가시화된 이미지를 저장할 경로
        
        """
        Path(write_path).mkdir(parents=True, exist_ok=True)

        gt_dict = self.__xmls2dict(self.__xmls_path['gt'], mode='gt')
        dt_dict = self.__xmls2dict(self.__xmls_path['dt'], mode='dt')

        for diagram in tqdm(dt_dict.keys(), f"Visualizing '{cls}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)

            for dt_item in dt_dict[diagram]:
                dt_bbox = dt_item['bndbox']
                dt_points = self.__dict2points(dt_bbox)
                dt_cls = dt_item['class']
                if dt_cls == cls or cls == 'total':
                    for num in range(4):
                        cv2.line(vis_img, dt_points[(num + 0) % 4], dt_points[(num + 1) % 4], (0, 0, 255), 4)

            for gt_item in gt_dict[diagram]:
                gt_bbox = gt_item['bndbox']
                gt_points = self.__dict2points(gt_bbox)
                gt_cls = gt_item['class']
                if gt_cls == cls or cls == 'total':
                    for num in range(4):
                        cv2.line(vis_img, gt_points[(num + 0) % 4], gt_points[(num + 1) % 4], (0, 255, 0), 2)

            vis_img_path = os.path.join(write_path, f"{diagram}.jpg")
            cv2.imwrite(vis_img_path, vis_img)

# pipeline

gt_imgs_path = 'D:\\Data\\PNID_RAW\\Drawing\\JPG_123'
gt_xmls_path = 'D:\\Data\\xml2eval\\GT_xmls_first_year_123'
dt_xmls_path = 'D:\\Data\\xml2eval\\DT_xmls_first_year_123'
symbol_txt_path = 'D:\\Data\\SymbolClass_Class.txt'
large_symbol_txt_path = 'D:\\Data\\SymbolClass_Class_big.txt'
dump_path = 'D:\\Experiments\\Detections\\from_xml\\first_year_123_50'
visualize_path = 'D:\\Experiments\\Visualization\\from_xml\\first_year_123_50'

eval = evaluate_from_xml(
                gt_xmls_path=gt_xmls_path,
                dt_xmls_path=dt_xmls_path,
                symbol_txt_path=symbol_txt_path,
                large_symbol_txt_path=large_symbol_txt_path,
                iou_thr=0.5
                )
eval.dump(dump_path=dump_path, 
          symbol_type='total',
          cmp_degree=False,)
eval.visualize(gt_imgs_path, visualize_path, cls='total')