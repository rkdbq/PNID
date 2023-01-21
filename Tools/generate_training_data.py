import os
import random
import pickle
#from Data_Generator.generate_segmented_data import generate_segmented_data
from Data_Generator.generate_bigsize_data import generate_bigsize_data
from Common.symbol_io import read_symbol_txt
from Data_Generator.write_coco_annotation import write_coco_annotation

from Data_Generator.genetate_segmented_data_refactor import generate_segmented_data

from pathlib import Path

# 학습 데이터 생성 코드. 도면을 train/test/val로 나누고, 각 set의 이미지를 분할하여 sub_img들로 만들어 저장함
# 이때, train의 경우 심볼(또는 옵션에 따라 심볼+텍스트)가 존재하지 않는 도면은 저장하지 않음
# 단 test/val 도면의 경우 심볼이 존재하지 않아도 저장함

base_dir = "C:\\pnid\\"
drawing_dir = base_dir + "Drawing\\JPG"
drawing_segment_dir = "C:\\pnid\\drawing_seg\\"
symbol_xml_dir = base_dir + "SymbolXML"
text_xml_dir = base_dir + "TextXML"

# 저장 directory 만들어 주는 부분 필요
for train_val_test in ['train', 'val', 'test']:
    Path(drawing_segment_dir + f'/{train_val_test}').mkdir(parents=True, exist_ok=True)

val_drawings = ['26071-200-M6-052-00004', '26071-200-M6-052-00013', '26071-200-M6-052-00015', '26071-200-M6-052-00021',
                '26071-200-M6-052-00032', '26071-200-M6-052-00036', '26071-200-M6-052-00048', '26071-200-M6-052-00074',
                '26071-200-M6-052-00081', '26071-200-M6-052-00083', '26071-200-M6-052-00084', '26071-200-M6-052-00086',
                '26071-200-M6-052-00101', '26071-200-M6-052-00115', '26071-300-M6-053-00004', '26071-300-M6-053-00007',
                '26071-300-M6-053-00021', '26071-300-M6-053-00301', '26071-500-M6-059-00021', '26071-500-M6-059-00024']
test_drawings = ['26071-200-M6-052-00001', '26071-200-M6-052-00002','26071-200-M6-052-00003', '26071-200-M6-052-00005', #'26071-200-M6-052-00006', '26071-200-M6-052-00056',
                '26071-200-M6-052-00077', '26071-200-M6-052-00107', '26071-200-M6-052-00120', '26071-300-M6-053-00003',
                '26071-300-M6-053-00025', '26071-300-M6-053-00027', '26071-300-M6-053-00263', '26071-300-M6-053-00271',
                '26071-300-M6-053-00302', '26071-300-M6-053-00305', '26071-300-M6-053-00310', '26071-500-M6-059-00007',
                '26071-500-M6-059-00009', '26071-500-M6-059-00014', '26071-500-M6-059-00017', '26071-500-M6-059-00022']
ignore_drawing = []
train_drawings = [x.split(".")[0] for x in os.listdir(symbol_xml_dir)
                  if x.split(".")[0] not in test_drawings and
                  x.split(".")[0] not in val_drawings and
                  x.split(".")[0] not in ignore_drawing ]

symbol_txt_path = base_dir + "Hyundai_SymbolClass_Sym_Only.txt"

include_text_as_class = True # Text를 별도의 클래스로 포함할 것인지 {"text"}
include_text_orientation_as_class = True # 세로 문자열을 또다른 별도의 클래스로 포함할 것인지 {"text_rotated"},

segment_params = [800, 800, 300, 300] # width_size, height_size, width_stride, height_stride
drawing_resize_scale = 0.5 # predict_postprocess.py의 drawing_resize_scale와 같은 값 

symbol_dict = read_symbol_txt(symbol_txt_path, include_text_as_class, include_text_orientation_as_class)

train_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in train_drawings]
val_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in val_drawings]
test_xmls = [os.path.join(symbol_xml_dir, f"{x}.xml") for x in test_drawings]

# # Random Shuffle
# train_ratio = 0.9
#
# random.Random(1).shuffle(xml_paths_without_test)
# train_count = int(len(xml_paths_without_test)*train_ratio)
# train_xmls = xml_paths_without_test[0:train_count]
# val_xmls = xml_paths_without_test[train_count:]

val_annotation_data = generate_segmented_data(val_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir,
                                              symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "val")
write_coco_annotation(os.path.join(drawing_segment_dir,"val.json"), val_annotation_data, symbol_dict, segment_params)

train_annotation_data = generate_segmented_data(train_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir,
                                                symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "train")
write_coco_annotation(os.path.join(drawing_segment_dir,"train.json"), train_annotation_data, symbol_dict, segment_params)

test_annotation_data = generate_segmented_data(test_xmls, drawing_dir, drawing_segment_dir, segment_params, text_xml_dir,
                                               symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "test")
write_coco_annotation(os.path.join(drawing_segment_dir,"test.json"), test_annotation_data, symbol_dict, segment_params)

# val_annotation_data = generate_bigsize_data(val_xmls, drawing_dir, drawing_segment_dir, text_xml_dir,
#                                               symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "val")
# write_coco_annotation(os.path.join(drawing_segment_dir,"val.json"), val_annotation_data, symbol_dict, segment_params)

# train_annotation_data = generate_bigsize_data(train_xmls, drawing_dir, drawing_segment_dir, text_xml_dir,
#                                                 symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "train")
# write_coco_annotation(os.path.join(drawing_segment_dir,"train.json"), train_annotation_data, symbol_dict, segment_params)

# test_annotation_data = generate_bigsize_data(test_xmls, drawing_dir, drawing_segment_dir, text_xml_dir,
#                                                symbol_dict, include_text_as_class, include_text_orientation_as_class, drawing_resize_scale, "test")
# write_coco_annotation(os.path.join(drawing_segment_dir,"test.json"), test_annotation_data, symbol_dict, segment_params)