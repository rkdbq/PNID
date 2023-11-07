import os, math, cv2
import numpy as np
from xml.dom import minidom
import xml.etree.ElementTree as ET
from shapely import Polygon
from pathlib import Path
from tqdm import tqdm

class text_merge():
    """ 텍스트 병합을 수행한다.

    Arguments:
        annxmls_path: 2점 좌표 + 각도 형식으로 표현된 annotation xml 폴더 경로
    """
    def __init__(self, annxmls_path: str):
        self.__annxmls_path = annxmls_path
        self.__TWO_POINTS_FORMAT = 22 # 2점 좌표 + 각도 포맷
        self.__FOUR_POINTS_FORMAT = 44 # 4점 좌표 (+ 각도) 포맷

    def __cal_iof(self, remain_points: tuple, remove_points: tuple):
        coords = self.__list2points(remain_points)
        remain_rect = Polygon(coords)

        coords = self.__list2points(remove_points)
        remove_rect = Polygon(coords)

        iof = 0
        if remain_rect.intersects(remove_rect):
            intersection = remain_rect.intersection(remove_rect).area
            iof = intersection / remove_rect.area
        return iof
    
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
    
    def __dict2points(self, points: dict):
        """ 점 딕셔너리를 리스트로 변환
        
        """
        coords = [float(points['x1']), float(points['y1']), float(points['x2']), float(points['y2']), 
                  float(points['x3']), float(points['y3']), float(points['x4']), float(points['y4']),]
        coords = np.array([int(i) for i in coords])
        coords = coords.reshape(4,2)
        coords = coords.tolist()
        return coords
    
    def __four2two(self, bbox: dict):
        """ 4점 좌표 형식을 2점 좌표 형식으로 변환
        
        """
        result = {}
        bbox_two = {}
        bbox_four = {}

        for key, value in bbox.items():
            result[key] = value
        for coord, value in bbox['bndbox'].items():
            bbox_four[coord] = float(value)

        mid = {}
        mid['x'] = (bbox_four['x1'] + bbox_four['x3']) / 2
        mid['y'] = (bbox_four['y1'] + bbox_four['y3']) / 2

        point = (bbox_four['x1'], bbox_four['y1'])
        degree = float(bbox['degree'])
        pivot = (mid['x'], mid['y'])
        x, y = self.__rotate(point, -degree, pivot)
        bbox_two['xmin'] = str(x)
        bbox_two['ymin'] = str(y)

        point = (bbox_four['x3'], bbox_four['y3'])
        degree = float(bbox['degree'])
        pivot = (mid['x'], mid['y'])
        x, y = self.__rotate(point, -degree, pivot)
        bbox_two['xmax'] = str(x)
        bbox_two['ymax'] = str(y)

        result['bndbox'] = bbox_two

        return result
    
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

        point = (bbox_two['xmax'], bbox_two['ymin'])
        x, y = self.__rotate(point, degree, pivot)
        bbox_four['x2'] = str(x)
        bbox_four['y2'] = str(y)

        point = (bbox_two['xmax'], bbox_two['ymax'])
        x, y = self.__rotate(point, degree, pivot)
        bbox_four['x3'] = str(x)
        bbox_four['y3'] = str(y)

        point = (bbox_two['xmin'], bbox_two['ymax'])
        x, y = self.__rotate(point, degree, pivot)
        bbox_four['x4'] = str(x)
        bbox_four['y4'] = str(y)

        result['bndbox'] = bbox_four

        return result
    
    def __xml2dict(self, element):
        """ xml 파일을 집합으로 파싱
        
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
                        diagram = filename.split('.xml')[0]
                        result[diagram] = self.__xml2dict(root_element)
                        if mode == self.__TWO_POINTS_FORMAT:
                            four_bboxs = []
                            for bbox in result[diagram]['symbol_object']:
                                four_bboxs.append(self.__two2four(bbox))
                            result[diagram]['symbol_object'] = four_bboxs
                    except ET.ParseError as e:
                        print(f'Error parsing {file_path}: {e}')
        return result
    
    def __list2points(self, points: list):
        coords = points
        coords = np.array([int(i) for i in coords])
        coords = coords.reshape(4,2)
        coords = coords.tolist()
        return coords
    
    def __get_merged_text(self, remain_str: str, remove_str: str):
        remain_len = len(remain_str)
        remove_len = len(remove_str)

        common_len = 0
        for i in range(1, min(remain_len, remove_len) + 1):
            if remain_str[-i:] == remove_str[:i]:
                common_len = i
        
        merged_str = remain_str + remove_str[common_len:]
        return merged_str
    
    def __get_merged_bbox(self, remain_bbox: dict, remove_bbox: dict):
        result = []
        remain_points = [float(point) for point in remain_bbox[2:10]]
        remove_points = [float(point) for point in remove_bbox[2:10]]
        mid = {}
        coords = {}
        coords['x'] = (sum(remain_points[0::2])) / 4
        coords['y'] = (sum(remain_points[1::2])) / 4
        mid['remain'] = coords
        coords['x'] = (sum(remove_points[0::2])) / 4
        coords['y'] = (sum(remove_points[1::2])) / 4
        mid['remove'] = coords
        mid['merged'] = {'x': (mid['remain']['x'] + mid['remove']['x']) / 2,
                         'y': (mid['remain']['y'] + mid['remove']['y']) / 2,}
        
        remain_type = remain_bbox[0]
        result.append(remain_type)

        remain_cls = remain_bbox[1]
        remove_cls = remove_bbox[1]

        merged_cls = self.__get_merged_text(remain_cls, remove_cls)
        result.append(merged_cls)

        idx = 0
        for remain_point, remove_point in zip(remain_points, remove_points):
            diff = {}
            coord = 'x' if idx % 2 == 0 else 'y'
            diff['remain'] = abs(remain_point - mid['merged'][coord])
            diff['remove'] = abs(remove_point - mid['merged'][coord])
            idx += 1
            if diff['remain'] > diff['remove']:
                result.append(remain_point)
            else:
                result.append(remove_point)

        remain_isLarge = remain_bbox[10]
        result.append(remain_isLarge)

        remain_degree = remain_bbox[11]
        result.append(remain_degree)

        remain_flip = remain_bbox[12]
        result.append(remain_flip)

        return tuple(result)
    
    def __cmp_iof(self, ann: list, iof_thr: float, y_diff_thr: int, y_diff_iof_thr: float):
        remain_ann = set(ann)
        merged_flag = [False]*ann.__len__()
        merged_cnt = 0
        for remain_bbox in ann:
                for remove_bbox in ann:
                    if merged_flag[ann.index(remain_bbox)]:
                        continue
                    if merged_flag[ann.index(remove_bbox)]:
                        continue
                    if remain_bbox != remove_bbox:
                        remain_type = remain_bbox[0]
                        remove_type = remove_bbox[0]
                        remain_degree = remain_bbox[11]
                        remove_degree = remove_bbox[11]
                        if remain_type == 'text' and remove_type == 'text' and remain_degree == remove_degree:
                            remain_points = [float(point) for point in remain_bbox[2:10]]
                            remove_points = [float(point) for point in remove_bbox[2:10]]
                            remain_y = {}
                            remove_y = {}
                            remain_y['min'] = min(remain_points[1::2])
                            remain_y['max'] = max(remain_points[1::2])
                            remove_y['min'] = min(remove_points[1::2])
                            remove_y['max'] = max(remove_points[1::2])
                            ymin_diff = abs(float(remain_y['min']) - float(remove_y['min']))
                            ymax_diff = abs(float(remain_y['max']) - float(remove_y['max']))
                            iof = self.__cal_iof(remain_points, remove_points)
                            horizontal_intersect = ymin_diff < y_diff_thr and ymax_diff < y_diff_thr and iof > y_diff_iof_thr
                            if iof > iof_thr or horizontal_intersect:

                                merged_bbox = self.__get_merged_bbox(remain_bbox, remove_bbox)
                                if remove_bbox in remain_ann:
                                    remain_ann.remove(remove_bbox)
                                if remain_bbox in remain_ann:
                                    remain_ann.remove(remain_bbox)
                                remain_ann.add(merged_bbox)
                                merged_flag[ann.index(remain_bbox)] = True
                                merged_flag[ann.index(remove_bbox)] = True
                                merged_cnt += 1
        return remain_ann, merged_cnt
    
    def __merge(self, iof_thr: float, y_diff_thr: int, y_diff_iof_thr: float, xml_dir_path: str):
        result = {}
        anns = self.__xmls2dict(xml_dir_path, mode=self.__TWO_POINTS_FORMAT)
        for diagram, ann in tqdm(anns.items(), "Merging"):
            result[diagram] = {}
            annset = []
            for obj in ann['symbol_object']:
                type = obj['type']
                cls = obj['class'] if obj['class'] is not None else ''
                coords = [str(coord) for coord in obj['bndbox'].values()]
                isLarge = obj['isLarge'] if 'isLarge' in obj else 'n'
                degree = obj['degree'] if 'degree' in obj else '0'
                flip = obj['flip'] if 'flip' in obj else 'n'
                bbox = (type,) + (cls,) + tuple(coords) + (isLarge,) + (degree,) + (flip,)
                annset.append(bbox)

            merged_annset = annset
            while True:
                merged_annset, merged_cnt = self.__cmp_iof(
                    ann=list(merged_annset),
                    iof_thr=iof_thr,
                    y_diff_thr=y_diff_thr,
                    y_diff_iof_thr=y_diff_iof_thr,
                )
                if merged_cnt == 0:
                    break

            result[diagram]['symbol_object'] = []
            for obj in merged_annset:
                symbol_obj = {
                    'type': obj[0],
                    'class': obj[1],
                    'bndbox': {
                        'x1': obj[2],
                        'y1': obj[3],
                        'x2': obj[4],
                        'y2': obj[5],
                        'x3': obj[6],
                        'y3': obj[7],
                        'x4': obj[8],
                        'y4': obj[9],
                    },
                    'isLarge': obj[10],
                    'degree': obj[11],
                    'flip': obj[12],
                }
                symbol_obj = self.__four2two(symbol_obj)
                for coord, value in symbol_obj['bndbox'].items():
                    symbol_obj['bndbox'][coord] = str(round(float(value)))
                result[diagram]['symbol_object'].append(symbol_obj)
        return result
         
    def __dict2xml(self, element, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                sub_element = ET.SubElement(element, key) 
                self.__dict2xml(sub_element, value)
            elif isinstance(value, list):
                for item in value:
                    self.__dict2xml(element, {key: item})
            else:
                sub_element = ET.SubElement(element, key) 
                sub_element.text = str(value)

    def text_merge_from_xmls(self, out_xmls_path: str, iof_thr: float = 0.3, y_diff_thr: int = 5, y_diff_iof_thr: float = 0.1):
        """ 텍스트 바운딩 박스들을 병합
        
        Arguments:
            out_xmls_path: 텍스트 병합 후 xml 파일들을 저장할 경로
            iof_thr: 병합의 기준이 되는 IoF
            y_diff_thr: 추가 병합의 기준이 되는 y값 차
            y_diff_iof_thr: 추가 병합의 기준이 되는 IoF
        """
        Path(out_xmls_path).mkdir(parents=True, exist_ok=True)

        remain_anns = self.__merge(
            iof_thr=iof_thr,
            y_diff_thr=y_diff_thr,
            y_diff_iof_thr=y_diff_iof_thr,
            xml_dir_path=self.__annxmls_path,
        )
        for diagram, ann in tqdm(remain_anns.items(), "Writing"):
            root = ET.Element('annotation')
            self.__dict2xml(root, ann)
            to_xml_path = os.path.join(out_xmls_path, f"{diagram}.xml")
            
            self.__indent(root)
            tree = ET.ElementTree(root)
            tree.write(to_xml_path)

            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
            return xmlstr

    def __indent(self, elem, level=0):  # Tools.Common.pnid_xml.py
        """ XML의 들여쓰기 포함한 출력을 위한 함수

        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.__indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def visualize(self, gt_imgs_path: str, befxmls_path: str, aftxmls_path: str, out_imgs_path: str, type: str = 'text'):
        """ 바운딩 박스들을 가시화
        
        Arguments:
            gt_imgs_path: 원본 이미지 파일들의 경로
            befxmls_path: 텍스트 병합 전 xml 파일들이 저장된 경로
            aftxmls_path: 텍스트 병합 후 xml 파일들이 저장된 경로
            out_imgs_path: 출력 이미지 파일들을 저장할 경로
            type: 가시화 할 심볼 타입 (특정 심볼 또는 total)
        """
        Path(out_imgs_path).mkdir(parents=True, exist_ok=True)

        bef_dict = self.__xmls2dict(befxmls_path, mode=self.__TWO_POINTS_FORMAT)
        aft_dict = self.__xmls2dict(aftxmls_path, mode=self.__TWO_POINTS_FORMAT)

        for diagram in tqdm(aft_dict.keys(), f"Visualizing '{type}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)
            for aft_item in aft_dict[diagram]['symbol_object']:
                aft_bbox = aft_item['bndbox']
                aft_points = self.__dict2points(aft_bbox)
                aft_type = aft_item['type']
                if aft_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, aft_points[(num + 0) % 4], aft_points[(num + 1) % 4], (0, 0, 255), 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_color = (50, 50, 50)  # 텍스트 색상 (BGR 형식)
                    font_thickness = 2
                    cv2.putText(vis_img, aft_item['class'], tuple(aft_points[0]), font, font_scale, font_color, font_thickness)
            
            vis_img_path = os.path.join(out_imgs_path, f"{diagram}_aft.jpg")
            cv2.imwrite(vis_img_path, vis_img)
            
        for diagram in tqdm(bef_dict.keys(), f"Visualizing '{type}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)
            for bef_item in bef_dict[diagram]['symbol_object']:
                bef_bbox = bef_item['bndbox']
                bef_points = self.__dict2points(bef_bbox)
                bef_type = bef_item['type']
                if bef_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, bef_points[(num + 0) % 4], bef_points[(num + 1) % 4], (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_color = (50, 50, 50)  # 텍스트 색상 (BGR 형식)
                    font_thickness = 2
                    cv2.putText(vis_img, bef_item['class'], tuple(bef_points[0]), font, font_scale, font_color, font_thickness)

            vis_img_path = os.path.join(out_imgs_path, f"{diagram}_bef.jpg")
            cv2.imwrite(vis_img_path, vis_img)

        for diagram in tqdm(aft_dict.keys(), f"Visualizing '{type}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)
            for aft_item in aft_dict[diagram]['symbol_object']:
                aft_bbox = aft_item['bndbox']
                aft_points = self.__dict2points(aft_bbox)
                aft_type = aft_item['type']
                if aft_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, aft_points[(num + 0) % 4], aft_points[(num + 1) % 4], (0, 0, 255), 4)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_color = (50, 50, 50)  # 텍스트 색상 (BGR 형식)
                    font_thickness = 2
                    cv2.putText(vis_img, aft_item['class'], tuple(aft_points[0]), font, font_scale, font_color, font_thickness)

            for bef_item in bef_dict[diagram]['symbol_object']:
                bef_bbox = bef_item['bndbox']
                bef_points = self.__dict2points(bef_bbox)
                bef_type = bef_item['type']
                if bef_type == type or type == 'total':
                    for num in range(4):
                        cv2.line(vis_img, bef_points[(num + 0) % 4], bef_points[(num + 1) % 4], (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0
                    font_color = (50, 50, 50)  # 텍스트 색상 (BGR 형식)
                    font_thickness = 2
                    cv2.putText(vis_img, bef_item['class'], tuple(bef_points[0]), font, font_scale, font_color, font_thickness)
            
            vis_img_path = os.path.join(out_imgs_path, f"{diagram}_total.jpg")
            cv2.imwrite(vis_img_path, vis_img)

gt_imgs_path = 'D:\\Data\\raw\\PNID_DOTA_before_split\\test\\images'
annxmls_path = 'D:\\Data\\xml2eval\\DT_test_second_year_before_text_merge'

merge = text_merge(annxmls_path)

merged_annxmls_path = 'D:\\Experiments\\Text_Merge\\from_xml\\DT_test_final_merged'
merge.text_merge_from_xmls(
    out_xmls_path=merged_annxmls_path,
    # iof_thr=0.8,
    # y_diff_thr=,
    # y_diff_iof_thr=0.8,
)

visualize_path = 'D:\\Experiments\\Visualizations\\from_xml\\DT_xmls_test_after_merge'
merge.visualize(
    gt_imgs_path=gt_imgs_path, 
    befxmls_path=annxmls_path, 
    aftxmls_path=merged_annxmls_path, 
    out_imgs_path=visualize_path, 
    type='text'
)