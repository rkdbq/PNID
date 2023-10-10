# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:27:50 2023

@author: 이유정

4점 text -> 2점 + 각도 xml
- 태그 isLarge 추가
- tkinter 삭제
- 각도 보정 코드 삭제
- 코드 리팩토링
- read_four_point_txt 함수 수정(빈 요소 삭제 코드 추가)
- cal_degree(각도 계산) 코드 수정
- 각도 부호 바꿔서 출력

"""

import xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
import math
import os


def create_class_type_map(file_path):
    classTypeTxt = open(file_path, "r")
    contents = classTypeTxt.read()
    contentsList = contents.split('\n')
    class_type_map = {}
    for i in contentsList:
        tmp = ''.join(i)
        tmpList = tmp.split('|')
        class_type_map[tmpList[1]] = tmpList[0]
    class_type_map['text'] = 'text'
    
    return class_type_map


def read_four_point_txt(file_path):
    fourPointTxt = open(file_path, "r")
    contents = fourPointTxt.read()
    contentsList = contents.split('\n')
    
    def filter_empty(item):
        return item != ['']
    
    fourPoint = list(filter(filter_empty, [tmp.split(' ') for tmp in contentsList]))
    return fourPoint


def convert_points(fourPoint):
    def remove_text(sublist):
        return sublist[:-1]

    def convert_sublist(sublist):
        return [(float(sublist[i]), float(sublist[i+1])) for i in range(0, len(sublist), 2)]

    converted_points = []
    for sublist in fourPoint:
        sublist = remove_text(sublist)
        sublist = convert_sublist(sublist)
        converted_points.append(sublist)

    return converted_points

def extract_rectangle_properties(pts):
    def calculate_distance(x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance
    
    width = calculate_distance(pts[0][0], pts[0][1], pts[1][0], pts[1][1]) 
    height = calculate_distance(pts[1][0], pts[1][1], pts[2][0], pts[2][1])
    center_x = (pts[0][0] + pts[2][0]) / 2
    center_y = (pts[0][1] + pts[2][1]) / 2
    return width, height, center_x, center_y


def cal_degree(after, center):
    def calculate_offset(point, center):
        x = point[0] - center[0]
        y = point[1] - center[1]
        return x, y
    x, y = calculate_offset(after, center)
    
    radian = math.atan2(y, x);
    degree = math.degrees(radian)
    if degree != 0:
        degree *= -1
    return degree

def rbox2hbox(center_x, center_y, width, height):
    before_left_top_point = [int(center_x - (width / 2)), int(center_y - (height / 2))]
    before_right_top_point = [int(center_x + (width / 2)), int(center_y - (height / 2))]
    before_left_bottom_point = [int(center_x - (width / 2)), int(center_y + (height / 2))]
    before_right_bottom_point = [int(center_x + (width / 2)), int(center_y + (height / 2))]
    return before_left_top_point, before_right_top_point, before_right_bottom_point, before_left_bottom_point

def create_xml(class_type_map, fourPoint, objects_list, output_file_path):
    xmlRoot = ET.Element("annotation")

    for i in range(len(objects_list)):
        obj = objects_list[i]
        
        elementSymbol = ET.Element("symbol_object")
        elementType = ET.Element("type")
        elementType.text = class_type_map[fourPoint[i][-1]]
        elementClass = ET.Element("class")
        elementClass.text = fourPoint[i][-1]

        elementBndbox = ET.Element("bndbox")
        elementXmin = ET.Element("xmin")
        elementXmin.text = str(obj['point1'][0])
        elementYmin = ET.Element("ymin")
        elementYmin.text = str(obj['point1'][1])
        elementXmax = ET.Element("xmax")
        elementXmax.text = str(obj['point3'][0])
        elementYmax = ET.Element("ymax")
        elementYmax.text = str(obj['point3'][1])

        elementBndbox.append(elementXmin)
        elementBndbox.append(elementYmin)
        elementBndbox.append(elementXmax)
        elementBndbox.append(elementYmax)

        elementIsLarge = ET.Element("isLarge")
        elementIsLarge.text = "n"  

        elementDegree = ET.Element("degree")
        elementDegree.text = str(obj['angle'])

        elementFlip = ET.Element("flip")
        elementFlip.text = "n"

        elementEtc = ET.Element("etc")

        elementSymbol.append(elementType)
        elementSymbol.append(elementClass)
        elementSymbol.append(elementBndbox)
        elementSymbol.append(elementIsLarge)
        elementSymbol.append(elementDegree)
        elementSymbol.append(elementFlip)
        elementSymbol.append(elementEtc)

        xmlRoot.append(elementSymbol)

    xmlstr = xml.dom.minidom.parseString(ET.tostring(xmlRoot)).toprettyxml(indent="    ")

    with open(output_file_path, "w") as file:
        file.write(xmlstr)

    return output_file_path


#4점 텍스트 파일이 들어있는 폴더 선택
txt_folder = 'D:\\Experiments\\Detections\\per_diagram_txt_annfiles\\roi_trans\\results_not_title'
#class_type 매핑 텍스트 파일 선택
class_file = "D:\\Data\\raw\\PNID_RAW\\Hyundai_SymbolClass_Type.txt"
class_type_map = create_class_type_map(class_file)
print(class_type_map)
# 결과 xml을 저장할 폴더 경로
output_folder = 'D:\\Data\\xml2eval\\DT_xmls_not_title'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 폴더 내의 모든 txt 파일에 대해 처리
for txt_file in os.listdir(txt_folder):
    if not txt_file.endswith('.txt'):
        continue
    txt_file_path = os.path.join(txt_folder, txt_file)
    fourPoint = read_four_point_txt(txt_file_path)
    
    #4점 텍스트 파일에서 클래스 지우고, str -> float으로 변경
    converted_points = convert_points(fourPoint)
    converted_points = np.array(converted_points, dtype=np.float32)
    
    objects_list = []
    for pts in converted_points:
        width, height, center_x, center_y = extract_rectangle_properties(pts)
        point1, point2, point3, point4 = rbox2hbox(center_x, center_y, width, height)
        after = [(p1 + p4) / 2 for p1, p4 in zip(pts[1], pts[2])]
        # after = pts[0]
        angle = cal_degree(after, (center_x, center_y))
        
        objects_list.append({
            'point1': point1,
            'point3': point3,
            'angle': angle,
        })
    
    
    output_file_path = os.path.join(output_folder, txt_file)
    output_file_path = output_file_path.replace(".txt", ".xml")
    output_filename = create_xml(class_type_map, fourPoint, objects_list, output_file_path)
    print(output_filename)

    
    