import os, cv2
import numpy as np
from shapely import Polygon
from pathlib import Path
from tqdm import tqdm

class evaluate_from_txt():
    def __init__(self, gt_txts_path: str, dt_txts_path: str, symbol_txt_path: str, iou_thr: float = 0.8):
        self.__txts_path = {}
        self.__txts_path['gt'] = gt_txts_path
        self.__txts_path['dt'] = dt_txts_path
        self.__symbol_txt_path = symbol_txt_path
        self.__iou_thr = iou_thr

    def __classtxt2diagramtxt(self, cls: str, classtxt_path: str, diagramtxt_dir_path: str, confi_thr: float = 0.5):
        
        file = open(classtxt_path, 'r')
        for line in file:
            info = line.split()
            diagram = info[0]
            confi = float(info[1])
            points = [round(float(i)) for i in info[2:10]]
            if confi < confi_thr: continue
            annfile = open(os.path.join(diagramtxt_dir_path, f"{diagram}.txt"), 'a+')
            annfile.write(f"{' '.join(map(str, points))} {cls}\n")
            annfile.close()
        return
    
    def classtxts2diagramtxts(self, classtxt_dir_path: str, diagramtxt_dir_path: str):

        Path(diagramtxt_dir_path).mkdir(parents=True, exist_ok=True)

        for root, dirs, files in os.walk(classtxt_dir_path):
            for filename in tqdm(files, "Class to Diagram"):
                if filename.startswith('Task1') and filename.endswith('.txt'):
                    cls = filename.replace("Task1_", "").replace(".txt", "")
                    classtxt_path = os.path.join(root, filename)
                    self.__classtxt2diagramtxt(cls, classtxt_path, diagramtxt_dir_path)
        return 

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

    def __anntxt2dict(self, txt_path: str, split_word: str = ' '):
        """ anntxt 파일을 딕셔너리로 파싱
        
        """
        result = []
        file = open(txt_path, 'r')
        for line in file:
            words = line.split(split_word)
            num = words[0:8]
            cls = words[8].replace('\n', '')
            result.append([num, cls])
        return result

    def __anntxts2dict(self, txt_dir_path: str):
        """ anntxt 파일들을 딕셔너리로 파싱
        
        """
        result = {}
        for root, dirs, files in os.walk(txt_dir_path):
            for filename in files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(root, filename)
                    result[filename[0:22]] = self.__anntxt2dict(file_path)
        return result
    
    def __list2points(self, points: list):
        coords = points
        coords = np.array([int(i) for i in coords])
        coords = coords.reshape(4,2)
        coords = coords.tolist()
        return coords

    def __cal_iou(self, gt_points: list, dt_points: list):
        """ IoU 계산 (바운딩 박스가 회전되어 있으므로 shapely.Polygon 사용)

        """
        coords = self.__list2points(gt_points)
        gt_rect = Polygon(coords)

        coords = self.__list2points(dt_points)
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
            iou_thr: IoU Threshold

        Returns:
            precision: {도면 이름: {클래스 이름: {tp, dt}}, ..., total: {tp, dt}}}}
            recall: {도면 이름: {클래스 이름: {tp, gt}}, ..., total: {tp, gt}}}}
        
        """
        precision = {}
        recall = {}

        for diagram in tqdm(gt_dict.keys(), "Evaluating"):
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
            for gt_item in gt_dict[diagram]:
                gt_bbox = gt_item[0]
                gt_cls = gt_item[1]
                for dt_item in dt_dict[diagram]:
                    dt_bbox = dt_item[0]
                    dt_cls = dt_item[1]
                    if gt_cls == dt_cls:
                        cls = gt_cls
                        if cls not in symbol_dict:
                            continue
                        if cls not in tp:
                            tp[cls] = 0
                        iou = self.__cal_iou(gt_bbox, dt_bbox)
                        if iou > self.__iou_thr:
                            tp[cls] += 1
            for dt_item in dt_dict[diagram]:
                cls = dt_item[1]
                if cls not in symbol_dict:
                    continue
                if cls not in dt:
                    dt[cls] = 0
                dt[cls] += 1
            for gt_item in gt_dict[diagram]:
                cls = gt_item[1]
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

    def dump(self, dump_path: str, model_name: str = ""):
        """ Precision, Recall을 계산하여 txt 파일로 출력
        
        Arguments:
            dump_path: result.txt를 저장할 경로
        
        """
        Path(dump_path).mkdir(parents=True, exist_ok=True)

        gt_dict = self.__anntxts2dict(self.__txts_path['gt'])
        dt_dict = self.__anntxts2dict(self.__txts_path['dt'])

        symbol_dict = self.__txt2dict(self.__symbol_txt_path)

        precision, recall = self.__evaluate(gt_dict, dt_dict, symbol_dict)

        mean = {}
        mean['tp'] = 0
        mean['dt'] = 0
        mean['gt'] = 0

        result_file = open(f"{dump_path}\\{model_name}_result.txt", 'w')
        result_file.write(f"IoU Threshold: {self.__iou_thr}\n")
        
        for diagram in tqdm(gt_dict.keys(), "Writing"):
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

            for cls, num in symbol_dict.items():
                if cls in recall[diagram]:
                    tp = recall[diagram][cls]['tp']
                    gt = recall[diagram][cls]['gt']
                    result_file.write(f"class {num} (['{cls}']): {tp} / {gt}\n")
            
            result_file.write(f'\n')

        mean['precision'] = mean['tp'] / mean['dt'] if mean['dt'] != 0 else 0
        mean['recall'] = mean['tp'] / mean['gt'] if mean['gt'] != 0 else 0
        result_file.write(f"(mean precision, mean recall) = ({mean['precision']}, {mean['recall']})")

        result_file.close()
        return
    
    def visualize(self, gt_imgs_path: str, write_path: str, cls: str = 'text'):
        """ 바운딩 박스들을 가시화
        
        Arguments:
            gt_imgs_path: 원본 이미지의 경로
            write_path: 가시화된 이미지를 저장할 경로
        
        """
        Path(write_path).mkdir(parents=True, exist_ok=True)

        gt_dict = self.__anntxts2dict(self.__txts_path['gt'])
        dt_dict = self.__anntxts2dict(self.__txts_path['dt'])

        for diagram in tqdm(dt_dict.keys(), f"Visualizing '{cls}' Class"):
            gt_img_path = os.path.join(gt_imgs_path, f"{diagram}.jpg")
            vis_img = cv2.imread(gt_img_path)

            gt_items = gt_dict[diagram]
            dt_items = dt_dict[diagram]

            for dt_item in dt_items:
                dt_bbox = dt_item[0]
                dt_points = self.__list2points(dt_bbox)
                dt_cls = dt_item[1]
                if dt_cls == cls or cls == 'all':
                    for num in range(4):
                        cv2.line(vis_img, dt_points[(num + 0) % 4], dt_points[(num + 1) % 4], (0, 0, 255), 4)

            for gt_item in gt_items:
                gt_bbox = gt_item[0]
                gt_points = self.__list2points(gt_bbox)
                gt_cls = gt_item[1]
                if gt_cls == cls or cls == 'all':
                    for num in range(4):
                        cv2.line(vis_img, gt_points[(num + 0) % 4], gt_points[(num + 1) % 4], (0, 255, 0), 2)

            vis_img_path = os.path.join(write_path, f"{diagram}.jpg")
            cv2.imwrite(vis_img_path, vis_img)

# pipeline

classtxt_dir_path = ''
gt_imgs_path = 'D:\\Data\PNID_DOTA_before_split\\test\\images'
gt_anntxts_path = 'D:\\Data\PNID_DOTA_before_split\\test\\annfiles_123'
dt_anntxts_path = 'D:\\Experiments\\Detections\\Diagrams\\roi_trans\\annfiles_123'
symbol_txt_path = 'D:\\Data\\SymbolClass_Class.txt'
dump_path = 'D:\\Experiments\\Detections\\from_txt\\test_123'
visualize_path = 'D:\\Experiments\\Visualization\\from_txt\\test_123'

eval = evaluate_from_txt(
                gt_txts_path=gt_anntxts_path,
                dt_txts_path=dt_anntxts_path,
                symbol_txt_path=symbol_txt_path,
                )

eval.classtxts2diagramtxts(
    classtxt_dir_path=classtxt_dir_path,
    diagramtxt_dir_path=dt_anntxts_path,
)
# eval.dump(dump_path, 'roi_trans')
# eval.visualize(gt_imgs_path, visualize_path, cls = 'all')