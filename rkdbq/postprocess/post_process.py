import os
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

base_dir = "C://Codes//GitHub//PNID//rkdbq//postprocess//"
symbol_dict_dir = base_dir + "SymbolClass_Class.txt"
detected_base_dir = base_dir + "ORCNN_2//"
detected_dir = detected_base_dir + "test//annfiles//"
ground_truth_dir = base_dir + "PNID_DOTA//test//annfiles//"


confidence_score_threshold = 0.5
IoU_threshold = 0.5

def get_filenames(dirname):
    filenames = os.listdir(dirname)
    return filenames

def make_detected_file_directory(dir):
    """ 클래스 별 분류된 텍스트 파일을 도면 별 분류된 텍스트 파일로 변환하여 저장할 경로 생성
    """
    Path(dir).mkdir(parents=True, exist_ok=True)

def add_line_to_diagram(line, diagramDir, className):
    info = line.split()
    points = [round(float(i)) for i in info[2:10]]
    confidenceScore = info[1]
    if float(confidenceScore) < confidence_score_threshold: return
    annfile = open(diagramDir + info[0] + ".txt", "a")
    annfile.write(" ".join(map(str, points)) + " " + className + "\n")
    annfile.close()

def convert_class_to_diagram(filesDir, diagramDir):
    """ 클래스 별 분류된 텍스트 파일을 도면 별 분류된 텍스트 파일로 변환

    Arguments:
        filesDir (string): 클래스 별 분류된 텍스트 파일들의 상위 경로
        diagramDir (string): 도면 별 분류된 텍스트 파일이 저장될 경로

    Returns:
        None

    """
    for fileName in get_filenames(filesDir):
        if "Task1_" not in fileName: continue
        className = fileName.replace("Task1_", "").replace(".txt", "")
        curFile = open(filesDir + fileName, "r")
        for line in curFile:
            add_line_to_diagram(line, diagramDir, className)   

def calculate_IoU(gt, dt):
    gtRect = Polygon(gt)
    dtRect = Polygon(dt)
    IoU = gtRect.intersection(dtRect).area / gtRect.union(dtRect).area
    return IoU

def compare_gt_and_dt_rotated(gt, dt, iouThreshold): # list -> map으로 구현 변경 필요.
    matched = {}
    for gtValue in gt:
        gtPoints = np.array([int(i) for i in gtValue[0:8]])
        gtPoints = gtPoints.reshape(4,2)
        gtPoints = gtPoints.tolist()
        gtClass = gtValue[8]
        for dtValue in dt:
            dtPoints = np.array([int(i) for i in dtValue[0:8]])
            dtPoints = dtPoints.reshape(4,2)
            dtPoints = dtPoints.tolist()
            dtClass = dtValue[8]
            if gtClass != dtClass: continue
            if calculate_IoU(gtPoints, dtPoints) > iouThreshold:
                if gtClass in matched:
                    matched[gtClass] += 1
                else:
                    matched[gtClass] = 1                
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


def diagram_text_to_dic(diagramDir):
    """ 도면 별 분류된 텍스트 파일을 dic으로 파싱

    Arguments:
        diagramDir (string): 도면 별 분류된 텍스트 파일들의 상위 경로

    Returns:
        totalValue (dict): 도면 이름을 key로, box들을 value로 갖는 dict
    """
    dic = {}
    for fileName in get_filenames(diagramDir):
        lis = text_to_list(diagramDir + fileName)
        diagramName = fileName.replace(".txt", "")
        dic[diagramName] = lis
    return dic

def symbol_dict_text_to_dic(symbol_dict_dir):
    lis = text_to_list(symbol_dict_dir, "|")
    dic = {}
    for symbol in lis:
        dic[symbol[1].replace("\n", "")] = int(symbol[0])
    return dic

def total_value(dic):
    totalValue = 0
    for value in dic.values():
        totalValue += value
    return totalValue

def calculate_rotated_pr(gt_result, dt_result):
    """ 전체 test 도면에 대한 precision 및 recall 계산

    Arguments:
        gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
        dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict

    Returns:
        pr_result (dict): 도면 이름을 key로, Precision 및 recall 계산에 필요한 데이터를 value로 갖는 dict
    """

    pr_result = {}

    for diagram in gt_result.keys():
        tpBoxes = compare_gt_and_dt_rotated(gt_result[diagram], dt_result[diagram], IoU_threshold)
        gtBoxes = total_bounding_box(gt_result[diagram])
        dtBoxes = total_bounding_box(dt_result[diagram])

        pr_result[diagram] = [dtBoxes, gtBoxes, tpBoxes]
        print(f"Calculate precision & recall {diagram} done")

    return pr_result

def dump_rotated_pr_result(pr_result, symbol_dict = 0):
        """ PR 계산 결과를 파일로 출력. test 내에 존재하는 모든 도면에 대해 한 파일로 한꺼번에 출력함

        Arguments:
            pr_result (dict): 도면 이름을 key로, 각 도면에서의 PR 계산에 필요한 정보들(detected_num, gt_num 및 클래스별 gt/dt num)을 저장한 dict
            symbol_dict (dict): 심볼 이름을 key로, id를 value로 갖는 dict

        Returns:
            None
        """
        tpMean = 0
        gtMean = 0
        dtMean = 0

        for diagram in pr_result.keys():
            dtBoxes = pr_result[diagram][0]
            gtBoxes = pr_result[diagram][1]
            tpBoxes = pr_result[diagram][2]

            tpTotal = total_value(tpBoxes)
            gtTotal = total_value(gtBoxes)
            dtTotal = total_value(dtBoxes)

            tpMean += tpTotal
            gtMean += gtTotal
            dtMean += dtTotal

            resultFile = open(base_dir + "result.txt", "a")
            resultFile.write(f"test drawing : {diagram}----------------------------------\n")
            resultFile.write(f"total precision : {tpTotal} / {dtTotal} = {tpTotal / dtTotal}\n")
            resultFile.write(f"total recall : {tpTotal} / {gtTotal} = {tpTotal / gtTotal}\n")

            for key in gtBoxes.keys():
                if key not in tpBoxes:
                    tpBoxes[key] = 0
                    dtBoxes[key] = 0
                resultFile.write(f"{key} : {tpBoxes[key]} / {gtBoxes[key]}\n")

            resultFile.write("\n")
            resultFile.close()
        
        resultFile = open(base_dir + "result.txt", "a")
        resultFile.write(f"(mean precision, mean recall) = ({tpMean / dtMean}, {tpMean / gtMean})")

# example
make_detected_file_directory(detected_dir)
convert_class_to_diagram(detected_base_dir, detected_dir)

gt_result = diagram_text_to_dic(ground_truth_dir)
dt_result = diagram_text_to_dic(detected_dir)

pr_result = calculate_rotated_pr(gt_result, dt_result)

# symbol_dict = symbol_dict_text_to_dic(symbol_dict_dir)

dump_rotated_pr_result(pr_result)