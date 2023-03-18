import os
import pprint
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon

base_dir = "C://Codes//postprocess//"
detected_base_dir = base_dir + "ORCNN_2//"
detected_dir = detected_base_dir + "test//annfiles//"
ground_truth_dir = base_dir + "PNID_DOTA//test//annfiles//"

confidence_score_threshold = 0.5
IoU_threshold = 0.5

def getFilenames(dirname):
    filenames = os.listdir(dirname)
    return filenames

def makeDirectory():
    Path(detected_dir).mkdir(parents=True, exist_ok=True)

def addLineToDiagram(line, className):
    info = line.split()
    points = [round(float(i)) for i in info[2:10]]
    confidenceScore = info[1]
    if float(confidenceScore) < confidence_score_threshold: return
    annfile = open(detected_dir + info[0] + ".txt", "a")
    annfile.write(" ".join(map(str, points)) + " " + className + "\n")
    annfile.close()

def convertClassToDiagram():
    for fileName in getFilenames(detected_base_dir):
        if "Task1_" not in fileName: continue
        className = fileName.replace("Task1_", "").replace(".txt", "")
        curFile = open(detected_base_dir + fileName, "r")
        for line in curFile:
            addLineToDiagram(line, className)   

def calculateIoU(gt, dt):
    gtRect = Polygon(gt)
    dtRect = Polygon(dt)
    IoU = gtRect.intersection(dtRect).area / gtRect.union(dtRect).area
    return IoU

def compare_gt_and_dt_rotated(gt, dt, iouThreshold):
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
    symbols = []
    dt = open(dir, "r")
    for line in dt:
        info = line.split()
        symbols.append(info)
    return symbols

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

    resultFile.write("test drawing :" + fileName.replace(".txt", "") + "----------------------------------" + "\n")
    resultFile.write("total precision :" + str(tpTotal) + "/" + str(dtTotal) + "=" + str(precision(tpBoxes, dtBoxes)) + "\n")
    resultFile.write("total recall :" + str(tpTotal) + "/" + str(gtTotal) + "=" + str(recall(tpBoxes, gtBoxes)) + "\n")

    for key in gtBoxes.keys():
        if key not in tpBoxes:
            tpBoxes[key] = 0
            dtBoxes[key] = 0
        resultFile.write(str(key) +":" + str(tpBoxes[key]) + "/" + str(gtBoxes[key]) + "\n")

    resultFile.write("\n")
    resultFile.close()

    return [tpTotal, dtTotal, tpTotal, gtTotal]

def writeResult(filesDir, resultDir):
    mean = []
    for fileName in getFilenames(filesDir):
        mean += addLineToResult(fileName, resultDir)

    resultFile = open(resultDir + "result.txt", "a")
    resultFile.write("mean precision, mean recall) = (" + str(precision(mean[0], mean[1])) + "," + str(recall(mean[2], mean[3])) + ")")

makeDirectory()
convertClassToDiagram()
print("write result.txt file...")
writeResult(ground_truth_dir, base_dir)
print("write result.txt file done!")