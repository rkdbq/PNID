import os
import numpy as np
from shapely import Polygon
from pathlib import Path
from tqdm import tqdm

class text_merge():
    def __init__(self, anntxts_path: str, iof_thr: float = 0.8):
        self.__anntxts_path = anntxts_path
        self.__iof_thr = iof_thr

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
    
    def __list2points(self, points: list):
        coords = points
        coords = np.array([int(i) for i in coords])
        coords = coords.reshape(4,2)
        coords = coords.tolist()
        return coords
    
    def __ann2dict(self, ann_path: str, split_word: str = ' '):
        """ ann 파일을 집합으로 파싱
        
        """
        result = set()
        file = open(ann_path, 'r')
        for line in file:
            words = line.split(split_word)
            words[8] = words[8].replace('\n', '')
            result.add(tuple(words))
        return result
    
    def __anns2dict(self, ann_dir_path: str):
        """ ann 파일들을 딕셔너리로 파싱
        
        """
        result = {}
        for root, dirs, files in os.walk(ann_dir_path):
            for filename in files:
                if filename.endswith('.txt'):
                    diagram = filename[0:22]
                    ann_path = os.path.join(root, filename)
                    result[diagram] = self.__ann2dict(ann_path)
        return result
    
    def __cmp_iof(self, ann: set):
        remain_ann = set.copy(ann)
        for remain_bbox in ann:
            for remove_bbox in ann:
                if remain_bbox != remove_bbox:
                    remain_cls = remain_bbox[8]
                    remove_cls = remove_bbox[8]
                    if remain_cls == 'text' and remove_cls == 'text':
                        remain_points = remain_bbox[0:8]
                        remove_points = remove_bbox[0:8]
                        iof = self.__cal_iof(remain_points, remove_points)
                        if iof > self.__iof_thr:
                            if remove_bbox in remain_ann:
                                remain_ann.remove(remove_bbox)
        return remain_ann
    
    def __merge(self, ann_dir_path: str):
        result = {}
        anns = self.__anns2dict(ann_dir_path)
        for diagram, ann in tqdm(anns.items(), "Merging"):
            result[diagram] = self.__cmp_iof(ann)
        return result
    
    def write_ann(self, write_path: str):
        Path(write_path).mkdir(parents=True, exist_ok=True)

        remain_anns = self.__merge(self.__anntxts_path)
        for diagram, ann in tqdm(remain_anns.items(), "Writing"):
            result_file = open(f"{write_path}/{diagram}.txt", 'a')
            for bbox in ann:
                len = bbox.__len__()
                if len == 10:
                    bbox = bbox[0:9]
                for element in bbox:
                    result_file.write(f"{element}")
                    if bbox.index(element) != 8: result_file.write(" ")
                result_file.write("\n")
            result_file.close()
        return

# pipeline

dt_anntxts_path = 'D:\\Experiments\\Detections\\Diagrams\\roi_trans\\annfiles'
write_path = 'D:\\Experiments\\Text_Merge\\roi_trans\\iof_40'

merge = text_merge(dt_anntxts_path, iof_thr=0.4).write_ann(write_path)