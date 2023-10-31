import os, cv2, math, sys
import xml.etree.ElementTree as ET
import numpy as np
from shapely import Polygon
from pathlib import Path
from tqdm import tqdm

class evaluate_from_xml():
    """ Precision 및 Recall을 측정한다.

    Arguments:
        gt_xmls_path: 정답 xml 폴더 경로
        dt_xmls_path: 검출 xml 폴더 경로
        symbol_txt_path: 심볼 클래스 딕셔너리 텍스트 파일 경로 (e.g. 1|flange)
        large_symbol_txt_path: 큰 심볼 클래스 딕셔너리 텍스트 파일 경로 (e.g. 1|vertical_drum)
        iou_thr: TP 기준 threshold
        symbol_type: 측정할 심볼 클래스 ('total', 'small', 'large' 또는 특정 심볼 클래스)
    """
    def __init__(self, gt_xmls_path: str, dt_xmls_path: str, symbol_txt_path: str, large_symbol_txt_path: str, iou_thr: float = 0.8, symbol_type: str = 'total'):
        self.__xmls_path = {}
        self.__xmls_path['gt'] = gt_xmls_path
        self.__xmls_path['dt'] = dt_xmls_path
        self.__symbol_txt_path = {}
        self.__symbol_txt_path['total'] = symbol_txt_path
        self.__symbol_txt_path['large'] = large_symbol_txt_path
        self.__iou_thr = iou_thr
        self.__TWO_POINTS_FORMAT = 22
        self.__FOUR_POINTS_FORMAT = 44

        self.symbol_type = symbol_type

        self.gt_dict = self.__xmls2dict(self.__xmls_path['gt'], mode=self.__TWO_POINTS_FORMAT)
        self.dt_dict = self.__xmls2dict(self.__xmls_path['dt'], mode=self.__TWO_POINTS_FORMAT)

        self.symbol_dict = {}
        self.symbol_dict['total'] = self.__txt2dict(self.__symbol_txt_path['total'])
        self.symbol_dict['large'] = self.__txt2dict(self.__symbol_txt_path['large'])
        self.symbol_dict['small'] = self.__diff_dict(self.symbol_dict['total'], self.symbol_dict['large'])
        self.symbol_dict['text'] = {
            'text': 178
        }

        self.symbol_dict = self.symbol_dict[self.symbol_type]

        self.precision, self.recall = self.__evaluate(self.gt_dict, self.dt_dict, self.symbol_dict)

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

    def __xmls2dict(self, xml_dir_path: str, mode: int):
        """ xml 파일들을 딕셔너리로 파싱
        
        """
        result = {}
        for root, dirs, files in os.walk(xml_dir_path):
            for filename in tqdm(files, f"Parcing XMLs"):
                if filename.endswith('.xml'):
                    file_path = os.path.join(root, filename)
                    try:
                        tree = ET.parse(file_path)
                        root_element = tree.getroot()
                        diagram = filename[0:22]
                        result[diagram] = self.__xml2dict(root_element)['symbol_object']
                        if mode == self.__TWO_POINTS_FORMAT:
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

    def __evaluate(self, gt_dict: dict, dt_dict: dict, symbol_dict: dict):
        """ Precision, Recall 계산에 필요한 TP, DT, GT 카운팅
        
        Arguments:
            gt_dict: GT xml로부터 파싱된 딕셔너리
            dt_dict: DT xml로부터 파싱된 딕셔너리
            symbol_dict: Symbol txt로부터 파싱된 딕셔너리

        Returns:
            precision: {도면 이름: {클래스 이름: {tp, dt}}, ..., total: {tp, dt}}}}
            recall: {도면 이름: {클래스 이름: {tp, gt}}, ..., total: {tp, gt}}}} 
        
        """
        precision = {}
        recall = {}
        for diagram in tqdm(gt_dict.keys(), f"Evaluation"):
            precision[diagram] = {}
            precision[diagram]['total'] = {}
            precision[diagram]['total']['tp'] = 0
            precision[diagram]['total']['dt'] = 0
            recall[diagram] = {}
            recall[diagram]['total'] = {}
            recall[diagram]['total']['tp'] = 0
            recall[diagram]['total']['gt'] = 0
            if diagram not in dt_dict: 
                print(f'{diagram} is skipped. (NOT exist in detection xmls path)\n')
                continue
            
            # Counting tp, dt, gt for each classes
            tp = {}
            dt = {}
            gt = {}

            gt_matched = [False]*gt_dict[diagram].__len__()
            dt_matched = [False]*dt_dict[diagram].__len__()
            for gt_item in gt_dict[diagram]:
                for dt_item in dt_dict[diagram]:
                    gt_idx = gt_dict[diagram].index(gt_item)
                    dt_idx = dt_dict[diagram].index(dt_item)
                    if gt_matched[gt_idx] or dt_matched[dt_idx]:
                        continue
                    if gt_item['type'] == dt_item['type']:
                        if gt_item['type'] == 'text' or gt_item['class'] == dt_item['class']:
                            cls = gt_item['class'] if gt_item['type'] != 'text' else 'text'
                            if cls not in symbol_dict:
                                continue
                            if cls not in tp:
                                tp[cls] = 0
                            iou = self.__cal_iou(gt_item['bndbox'], dt_item['bndbox'])
                            if iou > self.__iou_thr:
                                tp[cls] += 1
                                gt_matched[gt_idx] = True
                                dt_matched[dt_idx] = True
                                
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
                
        return precision, recall

    def dump(self, dump_path: str):
        """ Precision, Recall을 계산하여 txt 파일로 출력
        
        Arguments:
            dump_path: result.txt를 저장할 경로
        
        """

        Path(dump_path).mkdir(parents=True, exist_ok=True)
        
        precision, recall = self.precision, self.recall
        mean = {}
        mean['tp'] = 0
        mean['dt'] = 0
        mean['gt'] = 0

        result_file = open(os.path.join(dump_path, f"result_{self.symbol_type}.txt"), 'w')
        result_file.write(f"Symbol Type: {self.symbol_type}\n")
        result_file.write(f"IoU Threshold: {self.__iou_thr}\n")
        
        for diagram in tqdm(self.gt_dict.keys(), "Writing"):
            result_file.write(f"\n")
            result_file.write(f'test drawing: {diagram}----------------------------------\n')
            print(f'Diagram: {diagram}')

            score = "precision"
            tp = precision[diagram]['total']['tp']
            dt = precision[diagram]['total']['dt']
            pr = tp / dt if dt != 0 else 0
            result_file.write(f"total {score}: {tp} / {dt} = {pr}\n")
            print(f'precision: {pr}')

            score = "recall"
            tp = recall[diagram]['total']['tp']
            gt = recall[diagram]['total']['gt']
            rc = tp / gt if gt != 0 else 0
            result_file.write(f"total {score}: {tp} / {gt} = {rc}\n")
            print(f'recall: {rc}')

            print(f'(p+r)/2: {(pr+rc)/2}\n')

            mean['tp'] += tp
            mean['dt'] += dt
            mean['gt'] += gt

            for cls, num in self.symbol_dict.items():
                if cls in recall[diagram]:
                    tp = recall[diagram][cls]['tp']
                    gt = recall[diagram][cls]['gt']

                    result = f"class {num} (['{cls}']): {tp} / {gt}"
                    result = result + '\n'
                    result_file.write(result)      

            result_file.write('\n')

        mean['precision'] = mean['tp'] / mean['dt'] if mean['dt'] != 0 else 0
        mean['recall'] = mean['tp'] / mean['gt'] if mean['gt'] != 0 else 0
        result_file.write(f"(mean precision, mean recall) = ({mean['precision']}, {mean['recall']})\n")
        print(f"Mean precision, recall, (p+r)/2: {mean['precision']}, {mean['recall']}, {(mean['precision'] + mean['recall'])/2}\n")

        result_file.close()
        return
    
    def visualize(self, gt_imgs_path: str, gt_xmls_path: str, dt_xmls_path: str, out_imgs_path: str, type: str = 'text'):
        """ 바운딩 박스들을 가시화
        
        Arguments:
            gt_imgs_path: 원본 이미지 파일들의 경로
            gt_xmls_path: GT xml 파일들이 저장된 경로
            dt_xmls_path: DT xml 파일들이 저장된 경로
            out_imgs_path: 출력 이미지 파일들을 저장할 경로
            type: 가시화 할 심볼 타입 (특정 심볼 또는 total)
        """
        Path(out_imgs_path).mkdir(parents=True, exist_ok=True)

        gt_dict = self.__xmls2dict(gt_xmls_path, mode=self.__TWO_POINTS_FORMAT)
        dt_dict = self.__xmls2dict(dt_xmls_path, mode=self.__TWO_POINTS_FORMAT)

        for diagram in tqdm(dt_dict.keys(), f"Visualizing '{type}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)
            for dt_item in dt_dict[diagram]:
                dt_bbox = dt_item['bndbox']
                dt_points = self.__dict2points(dt_bbox)
                dt_type = dt_item['type']
                if dt_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, dt_points[(num + 0) % 4], dt_points[(num + 1) % 4], (0, 0, 255), 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_color = (0, 0, 0)  # 텍스트 색상 (BGR 형식)
                    font_thickness = 1
                    cv2.putText(vis_img, dt_item['class'], tuple(dt_points[0]), font, font_scale, font_color, font_thickness)
            
            vis_img_path = os.path.join(out_imgs_path, f"{diagram}_dt.jpg")
            cv2.imwrite(vis_img_path, vis_img)
            
        for diagram in tqdm(dt_dict.keys(), f"Visualizing '{type}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)
            for gt_item in gt_dict[diagram]:
                gt_bbox = gt_item['bndbox']
                gt_points = self.__dict2points(gt_bbox)
                gt_type = gt_item['type']
                if gt_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, gt_points[(num + 0) % 4], gt_points[(num + 1) % 4], (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_color = (0, 0, 0)  # 텍스트 색상 (BGR 형식)
                    font_thickness = 1
                    cv2.putText(vis_img, gt_item['class'], tuple(gt_points[0]), font, font_scale, font_color, font_thickness)

            vis_img_path = os.path.join(out_imgs_path, f"{diagram}_gt.jpg")
            cv2.imwrite(vis_img_path, vis_img)

        for diagram in tqdm(dt_dict.keys(), f"Visualizing '{type}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)
            for dt_item in dt_dict[diagram]:
                dt_bbox = dt_item['bndbox']
                dt_points = self.__dict2points(dt_bbox)
                dt_type = dt_item['type']
                if dt_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, dt_points[(num + 0) % 4], dt_points[(num + 1) % 4], (0, 0, 255), 4)
            
            for gt_item in gt_dict[diagram]:
                gt_bbox = gt_item['bndbox']
                gt_points = self.__dict2points(gt_bbox)
                gt_type = gt_item['type']
                if gt_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, gt_points[(num + 0) % 4], gt_points[(num + 1) % 4], (0, 255, 0), 2)        
            
            vis_img_path = os.path.join(out_imgs_path, f"{diagram}_total.jpg")
            cv2.imwrite(vis_img_path, vis_img)

# pipeline
if __name__=='__main__':
    if len(sys.argv) != 4:
        print("Usage: python eval_from_xml.py {GT xml 폴더 경로} {DT xml 폴더 경로} {출력 파일 경로}")
        exit()

    symbol_txt_path = 'D:\\Data\\SymbolClass_Class.txt'
    large_symbol_txt_path = 'D:\\Data\\SymbolClass_Class_big.txt'

    dt_xmls_path = sys.argv[1]
    gt_xmls_path = sys.argv[2]
    eval = evaluate_from_xml(
        gt_xmls_path=gt_xmls_path,
        dt_xmls_path=dt_xmls_path,
        symbol_txt_path=symbol_txt_path,
        large_symbol_txt_path=large_symbol_txt_path,
        iou_thr=0.5,
        symbol_type='total'
        )

    dump_path = sys.argv[3]
    eval.dump(dump_path)