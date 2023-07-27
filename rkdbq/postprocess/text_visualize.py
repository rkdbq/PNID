from post_process import diagram_text_to_dic
import cv2
import numpy as np

images_dir = "/Users/rkdbg/Codes/GitHub/PNID/rkdbq/postprocess/test_images/images/"
detected_dir = "/Users/rkdbg/Codes/GitHub/PNID/rkdbq/postprocess/s2anet/iou80/test/annfiles/"
save_dir = "/Users/rkdbg/Codes/GitHub/PNID/rkdbq/postprocess/s2anet/vis/"

def draw_rectangle_and_save(input_path, output_path, bboxs):
    """
    이미지에 직사각형을 그리고 새로 저장하는 함수

    :param input_path: 입력 이미지 파일 경로
    :param output_path: 출력 이미지 파일 경로
    """
    # 이미지를 불러옵니다.
    diagram = cv2.imread(input_path)

    # 직사각형을 그립니다.
    for bbox in bboxs:
        if bbox[8] == 'text':
            # 네 점을 이어 사각형 그리기
            points = [(bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5]), (bbox[6], bbox[7])]
            height = abs(int(bbox[3]) - int(bbox[5]))
            # if height > 80:
            if True:
                points = np.array(points, np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(diagram, [points], isClosed=True, color=(0, 0, 255), thickness=4)

    # 이미지를 저장합니다.
    cv2.imwrite(output_path, diagram)
    print("saved")

dt_result = diagram_text_to_dic(detected_dir)

for diagram, bboxs in dt_result.items():
    diagram_path = f"{images_dir}{diagram}.jpg"
    save_path = f"{save_dir}{diagram}.jpg"
    print(diagram_path)
    print(save_path)
    draw_rectangle_and_save(diagram_path, save_path, bboxs)