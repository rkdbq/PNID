import os, math
import xml.etree.ElementTree as ET
from pathlib import Path

def txt2dict(txt_path: str, split_word: str = '|'):
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

def rotate(point: tuple, degree: float, pivot: tuple):
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

def two2four(bbox: dict):
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
        x, y = rotate(point, degree, pivot)
        bbox_four['x1'] = str(x)
        bbox_four['y1'] = str(y)   

        point = (bbox_two['xmin'], bbox_two['ymax'])
        x, y = rotate(point, degree, pivot)
        bbox_four['x2'] = str(x)
        bbox_four['y2'] = str(y)

        point = (bbox_two['xmax'], bbox_two['ymax'])
        x, y = rotate(point, degree, pivot)
        bbox_four['x3'] = str(x)
        bbox_four['y3'] = str(y)

        point = (bbox_two['xmax'], bbox_two['ymin'])
        x, y = rotate(point, degree, pivot)
        bbox_four['x4'] = str(x)
        bbox_four['y4'] = str(y)

        result['bndbox'] = bbox_four

        return result

def xml2dict(element):
        """ xml 파일을 딕셔너리로 파싱
        
        """
        result = {}
        for child in element:
            child_data = xml2dict(child)
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

def dict2xml(element, dictionary):
    for key, value in dictionary.items():
        # sub_element = ET.ElementTree(element)
        if key != 'symbol_objects': 
            sub_element = ET.SubElement(element, key)

        if isinstance(value, dict):
            dict2xml(sub_element, value)
        elif isinstance(value, list):
            for item in value:
                dict2xml(element, {'symbol_object': item})
        else:
            sub_element.text = str(value)

def merge_xml(from_xmls_path: str, to_xmls_path: str, type_dict_path: str):
    types = txt2dict(type_dict_path)

    from_xmls = {}
    from_xmls['symbol'] = {}
    from_xmls['text'] = {}

    to_xmls = {}

    xml_path = {}
    xml_path['symbol'] = os.path.join(from_xmls_path, 'SymbolXML')
    xml_path['text'] = os.path.join(from_xmls_path, 'TextXML')

    for root, dirs, files in os.walk(xml_path['symbol']):
        for filename in files:
            if filename.endswith('.xml'):
                file_path = os.path.join(root, filename)
                tree = ET.parse(file_path)
                root_element = tree.getroot()
                diagram = filename.split('.')[0]
                from_xmls['symbol'][diagram] = xml2dict(root_element)

    for root, dirs, files in os.walk(xml_path['text']):
        for filename in files:
            if filename.endswith('.xml'):
                file_path = os.path.join(root, filename)
                tree = ET.parse(file_path)
                root_element = tree.getroot()
                diagram = filename.split('.')[0]
                from_xmls['text'][diagram] = xml2dict(root_element)

    for diagram, from_xml in from_xmls['symbol'].items():
        to_xmls[diagram] = {
            #  'filename': f"{diagram}.jpg",
            #  'size': from_xml['size'],
             'symbol_objects': [],
        }
        for from_symbol in from_xml['object']:
            bndbox = from_symbol['bndbox']
            cls = from_symbol['name']
            to_symbol = {
                'type': types[cls] if cls in types else 'unspecified_symbol',
                'class': cls,
                'bndbox': {
                    'x1': bndbox['xmin'],
                    'y1': bndbox['ymin'],
                    'x2': bndbox['xmax'],
                    'y2': bndbox['ymin'],
                    'x3': bndbox['xmax'],
                    'y3': bndbox['ymax'],
                    'x4': bndbox['xmin'],
                    'y4': bndbox['ymax'],
                },
                'degree': '0.0',
                'flip': 'n',
            }
            to_xmls[diagram]['symbol_objects'].append(to_symbol)

    for diagram, from_xml in from_xmls['text'].items():
         for from_text in from_xml['symbol_object']:
              bndbox = from_text['bndbox']
              to_text = {
                'type': 'text',
                'class': from_text['class'],
                'bndbox': {
                    'x1': bndbox['xmin'],
                    'y1': bndbox['ymin'],
                    'x2': bndbox['xmax'],
                    'y2': bndbox['ymin'],
                    'x3': bndbox['xmax'],
                    'y3': bndbox['ymax'],
                    'x4': bndbox['xmin'],
                    'y4': bndbox['ymax'],
                },
                'degree': '0.0',
                'flip': 'n',
              }
              to_xmls[diagram]['symbol_objects'].append(to_text)

    for diagram, to_xml in to_xmls.items():
         root = ET.Element('annotation')
         dict2xml(root, to_xmls[diagram])
         to_xml_path = os.path.join(to_xmls_path, f"{diagram}.xml")
         tree = ET.ElementTree(root)
         tree.write(to_xml_path)
    return


from_xmls_path = "D:\\Data\\PNID_RAW_not_title"
to_xmls_path = "D:\\Data\\xml2eval\\GT_xmls_not_title"
symbol_dict_path = "D:\\Data\\PNID_RAW\\Hyundai_SymbolClass_Type.txt"

Path(to_xmls_path).mkdir(parents=True, exist_ok=True)
merge_xml(from_xmls_path, to_xmls_path, symbol_dict_path)