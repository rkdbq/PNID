import os, shutil, sys, argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import Polygon

model_name = "roi_trans_with_angle_123"

# base_dir = "C:\\Codes\\GitHub\\PNID\\rkdbq\\postprocess\\"
symbol_dict_dir = "C:\\Codes\\GitHub\\PNID\\rkdbq\\postprocess\\SymbolClass_Class.txt"

detected_base_dir = f"D:\\Experiments\\mmrotate\\{model_name}\\"
ground_truth_dir = "D:\\Data\\PNID_DOTA_before_split\\test\\annfiles_123\\"


parser = argparse.ArgumentParser()
parser.add_argument("--iou", default="0.5", help="IoU_threshold를 입력하세요.")
parser.add_argument("--total", default="True")
args = parser.parse_args()

confidence_score_threshold = 0.5
IoU_threshold = float(args.iou)
before_remove = True
is_total = True
if args.total == "False" or args.total == "false": is_total = False

# remove_iou_path = detected_base_dir + f"iou{int(IoU_threshold*100)}\\"
detected_dir = f"D:\\Experiments\\Detections\\Diagrams\\{model_name}"

pr_per_confidence = {} # {confidence score threshold, {class, precision}}

def get_filenames(dirname):
    filenames = os.listdir(dirname)
    return filenames

def make_detected_file_directory(dir, init):
    """ 클래스 별 분류된 텍스트 파일을 도면 별 분류된 텍스트 파일로 변환하여 저장할 경로 생성
    """
    # if init:
    #     remove_directory(remove_iou_path)
    Path(dir).mkdir(parents=True, exist_ok=True)

def remove_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"디렉토리 '{directory_path}' 경로 초기화 완료")
    except OSError as e:
        print(f"디렉토리 '{directory_path}' 경로 초기화 오류: {e}")

def add_line_to_diagram(line, diagram_dir, class_name, confidence_score_threshold):
    info = line.split()
    points = [round(float(i)) for i in info[2:10]]
    angle = str(float(info[10]))
    confidence_score = info[1]
    if float(confidence_score) < confidence_score_threshold: return
    annfile = open(diagram_dir + info[0] + ".txt", "a")
    annfile.write(" ".join(map(str, points)) + " " + angle + " " + class_name + "\n")
    annfile.close()

def convert_class_to_diagram(files_dir, diagram_dir, confidence_score_threshold = 0.5):
    """ 클래스 별 분류된 텍스트 파일을 도면 별 분류된 텍스트 파일로 변환

    Arguments:
        files_dir (string): 클래스 별 분류된 텍스트 파일들의 상위 경로
        diagram_dir (string): 도면 별 분류된 텍스트 파일이 저장될 경로

    Returns:
        None

    """
    for file_name in get_filenames(files_dir):
        if "Task1_" not in file_name: continue
        class_name = file_name.replace("Task1_", "").replace(".txt", "")
        cur_file = open(files_dir + file_name, "r")
        for line in cur_file:
            add_line_to_diagram(line, diagram_dir, class_name, confidence_score_threshold)

def calculate_IoU(gt, dt):
    gt_rect = Polygon(gt)
    dt_rect = Polygon(dt)
    IoU = gt_rect.intersection(dt_rect).area / gt_rect.union(dt_rect).area
    return IoU

def compare_gt_and_dt_rotated(gt, dt, iou_threshold, diagram):
    matched = {}
    for gt_value in gt:
        gt_points = np.array([int(i) for i in gt_value[0:8]])
        gt_points = gt_points.reshape(4,2)
        gt_points = gt_points.tolist()
        gt_class = gt_value[8]
        for dt_value in dt:
            dt_points = np.array([int(i) for i in dt_value[0:8]])
            dt_points = dt_points.reshape(4,2)
            dt_points = dt_points.tolist()
            dt_class = dt_value[8]
            if gt_class != dt_class: continue
            if calculate_IoU(gt_points, dt_points) > iou_threshold:
                annfile = open(f"{detected_dir}{diagram}.txt", "a")
                annfile.write(f"{dt_value[0]} {dt_value[1]} {dt_value[2]} {dt_value[3]} {dt_value[4]} {dt_value[5]} {dt_value[6]} {dt_value[7]} {dt_value[8]}\n")
                annfile.close()
                if gt_class in matched:
                    matched[gt_class] += 1
                else:
                    matched[gt_class] = 1                
    return matched

def total_bounding_box(lists):
    boxes = {}
    for value in lists:
        clss = value[8]
        if clss in boxes:
            boxes[clss] += 1
        else:
            boxes[clss] = 1
    return boxes

def text_to_list(dir, split_word = " "):
    lis = []
    text = open(dir, "r")
    if split_word == " ":
        for line in text:
            info = line.split()
            lis.append(info)     
    else:
        for line in text:
            info = line.split(split_word)
            lis.append(info)

    return lis

def diagram_text_to_dic(diagram_dir):
    """ 도면 별 분류된 텍스트 파일을 dic으로 파싱

    Arguments:
        diagram_dir (string): 도면 별 분류된 텍스트 파일들의 상위 경로

    Returns:
        total_val (dict): 도면 이름을 key로, box들을 value로 갖는 dict
    """
    dic = {}
    for file_name in get_filenames(diagram_dir):
        lis = text_to_list(diagram_dir + file_name)
        diagram_name = file_name.replace(".txt", "")
        dic[diagram_name] = lis
    return dic

def symbol_dict_text_to_dic(symbol_dict_dir):
    lis = text_to_list(symbol_dict_dir, "|")
    dic = {}
    for symbol in lis:
        dic[symbol[1].replace("\n", "")] = int(symbol[0])
    return dic

def total_value(dic):
    total_val = 0
    for value in dic.values():
        total_val += value
    return total_val

def calculate_rotated_pr(gt_result, dt_result):
    """ 전체 test 도면에 대한 precision 및 recall 계산

    Arguments:
        gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
        dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict

    Returns:
        pr_result (dict): 도면 이름을 key로, Precision 및 recall 계산에 필요한 데이터를 value로 갖는 dict
    """

    pr_result = {}

    for diagram in tqdm(gt_result.keys(), desc="precision and recall calculations"):
        # get() 메서드를 사용하여 키에 해당하는 값이 없을 경우 빈 리스트를 반환하도록 함
        tp_boxes = compare_gt_and_dt_rotated(gt_result.get(diagram, []), dt_result.get(diagram, []), IoU_threshold, diagram)
        gt_boxes = total_bounding_box(gt_result.get(diagram, []))
        dt_boxes = total_bounding_box(dt_result.get(diagram, []))

        pr_result[diagram] = [dt_boxes, gt_boxes, tp_boxes]

    return pr_result

def dump_rotated_pr_result(pr_result, symbol_dict = 0, confidence_score_threshold = 0.5, mAP = 0):
        """ PR 계산 결과를 파일로 출력. test 내에 존재하는 모든 도면에 대해 한 파일로 한꺼번에 출력함

        Arguments:
            pr_result (dict): 도면 이름을 key로, 각 도면에서의 PR 계산에 필요한 정보들(detected_num, gt_num 및 클래스별 gt/dt num)을 저장한 dict
            symbol_dict (dict): 심볼 이름을 key로, id를 value로 갖는 dict

        Returns:
            None
        """
        tp_mean = 0
        gt_mean = 0
        dt_mean = 0

        ap_mean = 0

        if is_total:
            result_file = open(f"{detected_dir}_result.txt", "a")
        else:
            result_file = open(f"{detected_dir}_123_result.txt", "a")
        
        result_file.write(f"Model : {model_name}\n")
        result_file.write(f"IoU threshold : {IoU_threshold}\n")
        result_file.write(f"Confidence Score threshold : {confidence_score_threshold}\n")
        result_file.write("\n")

        total_dt_boxes = {}
        total_tp_boxes = {}

        for diagram in pr_result.keys():
            dt_boxes = pr_result[diagram][0]
            gt_boxes = pr_result[diagram][1]
            tp_boxes = pr_result[diagram][2]

            tp_total = total_value(tp_boxes)
            gt_total = total_value(gt_boxes)
            dt_total = total_value(dt_boxes)

            tp_mean += tp_total
            gt_mean += gt_total
            dt_mean += dt_total

            result_file.write(f"test drawing : {diagram}----------------------------------\n")
            result_file.write(f"total precision : {tp_total} / {dt_total} = {tp_total / dt_total if dt_total != 0 else 0}\n")
            result_file.write(f"total recall : {tp_total} / {gt_total} = {tp_total / gt_total}\n")

            for key in symbol_dict.keys():
                if key not in tp_boxes:
                    continue
                if key not in total_tp_boxes:
                    total_tp_boxes[key] = 0
                if key not in total_dt_boxes:
                    total_dt_boxes[key] = 0

                total_tp_boxes[key] += tp_boxes[key]
                total_dt_boxes[key] += dt_boxes[key]                

            for key in symbol_dict.keys():
                if key not in gt_boxes:
                    continue
                if key not in tp_boxes:
                    tp_boxes[key] = 0
                    dt_boxes[key] = 0
                
                result_file.write(f"class {symbol_dict[key]} (['{key}']) : {tp_boxes[key]} / {gt_boxes[key]}\n")
                if dt_boxes[key] == 0: 
                    continue
                ap_mean = ap_mean + (tp_boxes[key] / dt_boxes[key]) / len(symbol_dict.keys())

            result_file.write("\n")
        
        result_file.write(f"(mean precision, mean recall, mAP) = ({tp_mean / dt_mean if dt_mean != 0 else 0}, {tp_mean / gt_mean if gt_mean != 0 else 0}, {mAP})")
        result_file.close()

        for cls in total_tp_boxes.keys():
            pr = {cls: total_tp_boxes.get(key, 0) / total_dt_boxes.get(key, 1)}
            pr_per_confidence[confidence_score_threshold] = pr

# use

# make_detected_file_directory(detected_iou_dir, before_remove)
# convert_class_to_diagram(detected_base_dir, detected_iou_dir, confidence_score_threshold)

# gt_result = diagram_text_to_dic(ground_truth_dir)
# dt_result = diagram_text_to_dic(detected_iou_dir)

# pr_result = calculate_rotated_pr(gt_result, dt_result)

# symbol_dict = symbol_dict_text_to_dic(symbol_dict_dir)

# dump_rotated_pr_result(pr_result, symbol_dict, confidence_score_threshold)