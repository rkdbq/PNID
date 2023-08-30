from post_process import diagram_text_to_dic, convert_class_to_diagram
from tqdm import tqdm
import cv2
import numpy as np
import datetime
from pathlib import Path

current_datetime = datetime.datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d_%H%M%S")

model_name = "roi_trans_with_angle_123"

test_dir = "D:\\Data\\PNID_DOTA_before_split\\test\\"
images_dir =  test_dir + "images\\"
gt_dir = test_dir + "annfiles\\"
dt_base_dir = f"D:\\Experiments\\mmrotate\\{model_name}\\"
diagram_dir = f"D:\\Experiments\\Detections\\Diagrams\\{model_name}\\{datetime_string}\\"
vis_dir = f"D:\\Experiments\\Detections\\Text_visualize\\{model_name}\\{datetime_string}\\"

def draw_rectangle_and_save(input_path, output_path, gt_bboxs, dt_bboxs = 0):
    """
    이미지에 직사각형을 그리고 새로 저장하는 함수

    :param input_path: 입력 이미지 파일 경로
    :param output_path: 출력 이미지 파일 경로
    """
    # 이미지를 불러옵니다.
    diagram = cv2.imread(input_path)
    if diagram is None:
        return

    if dt_bboxs != 0:
        for bbox in dt_bboxs:
            if bbox[8] == 'text':
                # 네 점을 이어 사각형 그리기
                bbox = [int(float(element)) for element in bbox[0:8]]
                points = [(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]), (bbox[6], bbox[7])]

                points = np.array(points, np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(diagram, [points], isClosed=True, color=(0, 0, 255), thickness=4)
    
    for bbox in gt_bboxs:
        if bbox[8] == 'text':
            # 네 점을 이어 사각형 그리기
            bbox = [int(float(element)) for element in bbox[0:8]]
            points = [(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]), (bbox[6], bbox[7])]

            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(diagram, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 이미지를 저장합니다.
    cv2.imwrite(output_path, diagram)

Path(vis_dir).mkdir(parents=True, exist_ok=True)
Path(diagram_dir).mkdir(parents=True, exist_ok=True)
convert_class_to_diagram(dt_base_dir, diagram_dir, 0.8)
# gt_result = diagram_text_to_dic(gt_dir)
# dt_result = diagram_text_to_dic(diagram_dir)

# for diagram in tqdm(gt_result.keys(), desc="Text visualization in patch"):
#     diagram_path = f"{images_dir}{diagram}.jpg"
#     save_path = f"{vis_dir}{diagram}.jpg"
#     draw_rectangle_and_save(diagram_path, save_path, gt_result[diagram], dt_result[diagram])