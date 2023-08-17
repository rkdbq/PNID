from post_process import diagram_text_to_dic
from tqdm import tqdm
import cv2
import numpy as np

images_dir = "C:\\Codes\\GitHub\\PNID\\rkdbq\\postprocess\\PNID_DOTA_before_split\\test\\images\\"
gt_dir = "C:\\Codes\\GitHub\\PNID\\rkdbq\\postprocess\\PNID_DOTA_before_split\\test\\annfiles\\"
dt_dir = "C:\\Codes\\GitHub\\PNID\\rkdbq\\postprocess\\roi_trans\\iou50\\test\\annfiles\\"
save_dir = "C:\\Codes\\GitHub\\PNID\\rkdbq\\postprocess\\roi_trans\\vis_iou50_530\\"

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

gt_result = diagram_text_to_dic(gt_dir)
dt_result = diagram_text_to_dic(dt_dir)

for diagram in tqdm(gt_result.keys(), desc="Text visualization in patch"):
    diagram_path = f"{images_dir}{diagram}.jpg"
    save_path = f"{save_dir}{diagram}.jpg"
    draw_rectangle_and_save(diagram_path, save_path, gt_result[diagram], dt_result[diagram])