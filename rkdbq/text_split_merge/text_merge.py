import os
from shapely import Polygon
from pathlib import Path
from tqdm import tqdm

class text_merge():
    def __init__(self, annfiles_path: str, iof_thr: float = 0.8):
        self.__annfiles_path = annfiles_path
        self.__iof_thr = iof_thr

    def __cal_iof(self, remain_points: tuple, remove_points: tuple):
        coords = [(remain_points[0], remain_points[1]), 
                  (remain_points[2], remain_points[3]), 
                  (remain_points[4], remain_points[5]), 
                  (remain_points[6], remain_points[7]),]
        remain_rect = Polygon(coords)

        coords = [(remove_points[0], remove_points[1]), 
                  (remove_points[2], remove_points[3]), 
                  (remove_points[4], remove_points[5]), 
                  (remove_points[6], remove_points[7]),]
        remove_rect = Polygon(coords)

        intersection = remain_rect.intersection(remove_rect).area
        iof = intersection / remove_rect.area
        return iof
    
    def __ann2dict(self, ann_path: str, split_word: str = ' '):
        """ ann 파일을 집합으로 파싱
        
        """
        result = set()
        file = open(ann_path, 'r')
        for line in file:
            words = line.split(split_word)
            result.add(tuple(words))
        return result
    
    def __anns2dict(self, ann_dir_path: str):
        """ ann 파일들을 딕셔너리로 파싱
        
        """
        result = {}
        for root, dirs, files in os.walk(ann_dir_path):
            for filename in files:
                if filename.endswith('.txt'):
                    ann_path = os.path.join(root, filename)
                    try:
                        result[filename[0:22]] = self.__ann2dict(ann_path)
                    except ValueError as e:
                        print(f'Error: {e}')
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
        for diagram, ann in tqdm(anns.items(), f"Merging"):
            result[diagram] = self.__cmp_iof(ann)
        return result
    
    def write_ann(self, write_path: str):
        Path(write_path).mkdir(parents=True, exist_ok=True)

        remain_anns = self.__merge(self.__annfiles_path)
        for diagram, ann in remain_anns.items():
            result_file = open(f"{write_path}/{diagram}.txt", 'a')
            for bbox in ann:
                for element in bbox:
                    result_file.write(f"{element}")
                    if bbox.index(element) != 9: result_file.write(" ")
            result_file.close()
        return

# pipeline

ann_dir_path = "D:\\Data\\PNID_DOTA_before_split\\test\\annfiles"
write_path = "D:\\Experiments\\Text_Merge\\roi_trans_merged"

merge = text_merge(ann_dir_path, iof_thr=0.3).write_ann(write_path)