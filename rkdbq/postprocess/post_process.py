import os
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

base_dir = "//Users//rkdbg//Codes//GitHub//PNID//rkdbq//postprocess//"
detected_base_dir = base_dir + "ORCNN_2//"
detected_dir = detected_base_dir + "test//annfiles//"
ground_truth_dir = base_dir + "PNID_DOTA//test//annfiles//"

confidence_score_threshold = 0.5
IoU_threshold = 0.5

def getFilenames(dirname):
    filenames = os.listdir(dirname)
    return filenames

def makeDirectory(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)

def addLineToDiagram(line, diagramDir, className):
    info = line.split()
    points = [round(float(i)) for i in info[2:10]]
    confidenceScore = info[1]
    if float(confidenceScore) < confidence_score_threshold: return
    annfile = open(diagramDir + info[0] + ".txt", "a")
    annfile.write(" ".join(map(str, points)) + " " + className + "\n")
    annfile.close()

def convertClassToDiagram(filesDir, diagramDir):
    for fileName in getFilenames(filesDir):
        if "Task1_" not in fileName: continue
        className = fileName.replace("Task1_", "").replace(".txt", "")
        curFile = open(filesDir + fileName, "r")
        for line in curFile:
            addLineToDiagram(line, diagramDir, className)   

def calculateIoU(gt, dt):
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
            if calculateIoU(gtPoints, dtPoints) > iouThreshold:
                if gtClass in matched:
                    matched[gtClass] += 1
                else:
                    matched[gtClass] = 1                
    return matched

def totalBoundingBox(lists):
    boxes = {}
    for value in lists:
        clss = value[8]
        if clss in boxes:
            boxes[clss] += 1
        else:
            boxes[clss] = 1
    return boxes

def textToList(dir):
    lis = []
    text = open(dir, "r")
    for line in text:
        info = line.split()
        lis.append(info)
    return lis

def textToDic(filesDir):
    for fileName in getFilenames(filesDir):
        lis = textToList(filesDir + fileName)
        dic = {}
        diagramName = fileName.replace(".txt", "")
        dic[diagramName] = lis
    return dic

def totalValue(dic):
    totalValue = 0
    for value in dic.values():
        totalValue += value
    return totalValue

def precision(tp, dt):
    return totalValue(tp) / totalValue(dt)

def recall(tp, gt):
    return totalValue(tp) / totalValue(gt)

def addLineToResult(fileName, dir):
    resultFile = open(dir + "result.txt", "a")

    tpBoxes = compare_gt_and_dt_rotated(textToList(ground_truth_dir + fileName), textToList(detected_dir + fileName), IoU_threshold)
    gtBoxes = totalBoundingBox(textToList(ground_truth_dir + fileName))
    dtBoxes = totalBoundingBox(textToList(detected_dir + fileName))

    tpTotal = totalValue(tpBoxes)
    gtTotal = totalValue(gtBoxes)
    dtTotal = totalValue(dtBoxes)

    diagramName = fileName.replace(".txt", "")
    resultFile.write(f"test drawing : {diagramName}----------------------------------\n")
    resultFile.write(f"total precision : {tpTotal} / {dtTotal} = {tpTotal / dtTotal}\n")
    resultFile.write(f"total recall : {tpTotal} / {gtTotal} = {tpTotal / gtTotal}\n")

    for key in gtBoxes.keys():
        if key not in tpBoxes:
            tpBoxes[key] = 0
            dtBoxes[key] = 0
        resultFile.write(f"{key} : {tpBoxes[key]} / {gtBoxes[key]}\n")

    resultFile.write("\n")
    resultFile.close()

    return [tpTotal, dtTotal, tpTotal, gtTotal]

def writeResult(filesDir, resultDir):
    mean = []
    for fileName in getFilenames(filesDir):
        mean += addLineToResult(fileName, resultDir)

    meanPrecision = mean[0] / mean[1]
    meanRecall = mean[2] / mean[3]

    resultFile = open(resultDir + "result.txt", "a")
    resultFile.write(f"(mean precision, mean recall) = ({meanPrecision}, {meanRecall})")

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
        gtBoxes = totalBoundingBox(gt_result[diagram])
        dtBoxes = totalBoundingBox(dt_result[diagram])

        pr_result[diagram] = [dtBoxes, gtBoxes, tpBoxes]

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

            tpTotal = totalValue(tpBoxes)
            gtTotal = totalValue(gtBoxes)
            dtTotal = totalValue(dtBoxes)

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


makeDirectory(detected_dir)
convertClassToDiagram(detected_base_dir, detected_dir)

gt_result = textToDic(ground_truth_dir)
dt_result = textToDic(detected_dir)
pr_result = calculate_rotated_pr(gt_result, dt_result)

print("write result.txt file...")
dump_rotated_pr_result(pr_result)
print("write result.txt file done!")