import os, sys, io
import numpy as np
import cv2 as cv
from collections import defaultdict
from pycocotools import coco, cocoeval
from pathlib import Path


from Common.coco_json import coco_json_write

class evaluate():
    """ Precision-Recall, AP 성능 계산 및 결과 Dump

    Arguments:
        output_dir (string) : dump 및 중간 과정에서 생성되는 COCO 형식의 dt, gt 데이터가 저장될 폴더
    """
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def compare_gt_and_dt(self, gt_result, dt_result, matching_iou_threshold):
        # TODO : 완전한 결과를 보려면 matching이 되지않은 dt의 정보도 전달되면 좋을듯
        """ GT와 DT 결과를 비교하여 매칭이 성공한 index들을 dict로 반환하는 함수

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            matching_iou_threshold (float): IOU > threshold이고 카테고리(class)가 같으면 매칭된 것으로 처리
        Returns:
            gt_to_dt_match_dict (dict): gt의 심볼 index를 key로, 매칭된 dt의 심볼 index를 value로 갖는 dict
            dt_to_gt_match_dict (dict): dt의 심볼 index를 key로, 매칭된 gt의 심볼 index를 value로 갖는 dict
        """
        gt_to_dt_match_dict = {}
        dt_to_gt_match_dict = {}

        for filename, annotations in dt_result.items():
            gt_to_result_index_match_dict = {}
            result_to_gt_index_match_dict = {}

            gt_boxes = [x["bbox"] for x in gt_result[filename]]
            gt_boxes = np.array(gt_boxes)
            g_w = gt_boxes[:, 2]
            g_h = gt_boxes[:, 3]
            gt_boxes_area = g_w * g_h
            gt_boxes_class = [x["category_id"] for x in gt_result[filename]]
            gt_boxes_class = np.array(gt_boxes_class)

            boxes = np.array([x["bbox"] for x in annotations])
            classes = np.array([x["category_id"] for x in annotations])
            scores = np.array([x["score"] for x in annotations])
            result_boxes = np.zeros((boxes.shape[0], boxes.shape[1] + 2))
            result_boxes[:,0:4] = boxes
            result_boxes[:,4] = classes
            result_boxes[:,5] = scores

            result_boxes_score = result_boxes[:, -1]
            result_boxes_score_sorted_index = (-result_boxes_score).argsort()
            result_boxes_score_sorted = result_boxes[result_boxes_score_sorted_index]

            for result_index in range(result_boxes_score_sorted.shape[0]):
                result_box, result_box_class = result_boxes_score_sorted[result_index, :-2], result_boxes_score_sorted[
                    result_index, -2]

                same_class_gt_box_index = (result_box_class == gt_boxes_class)
                if np.any(same_class_gt_box_index) == False:
                    continue

                r_w = result_box[2]
                r_h = result_box[3]
                result_box_area = r_w * r_h

                intersection_x1 = np.maximum(result_box[0], gt_boxes[:, 0])
                intersection_y1 = np.maximum(result_box[1], gt_boxes[:, 1])
                intersection_x2 = np.minimum(result_box[0] + result_box[2], gt_boxes[:, 0] + gt_boxes[:, 2])
                intersection_y2 = np.minimum(result_box[1] + result_box[3], gt_boxes[:, 1] + gt_boxes[:, 3])

                intersection_w = np.maximum(0, intersection_x2 - intersection_x1 + 1)
                intersection_h = np.maximum(0, intersection_y2 - intersection_y1 + 1)
                intersection = intersection_w * intersection_h

                iou = intersection / (gt_boxes_area + result_box_area - intersection)

                iou_threshold = matching_iou_threshold
                over_IOU_index = iou > iou_threshold
                iou_sorted_index = (-iou).argsort()

                iou_sorted_over_threshold_same_class_index = (over_IOU_index & same_class_gt_box_index)[iou_sorted_index]
                iou_sorted_same_class_over_iou_threshold_indexs = np.where(iou_sorted_over_threshold_same_class_index)[0]

                for iou_sorted_same_class_over_iou_threshold_index in iou_sorted_same_class_over_iou_threshold_indexs:
                    real_gt_index = iou_sorted_index[iou_sorted_same_class_over_iou_threshold_index]
                    real_result_index = result_boxes_score_sorted_index[result_index]

                    if real_gt_index not in gt_to_result_index_match_dict.keys():
                        gt_to_result_index_match_dict[real_gt_index] = real_result_index
                        result_to_gt_index_match_dict[result_index] = real_gt_index
                        break

            gt_to_dt_match_dict[filename] = gt_to_result_index_match_dict
            dt_to_gt_match_dict[filename] = result_to_gt_index_match_dict

        return gt_to_dt_match_dict, dt_to_gt_match_dict
    
    def label_cropped_image(self, gt_result, dt_result, gt_to_dt_match_dict, symbol_dict, imgs_path):
        """ Recognition 결과를 파일로 출력. test 내에 존재하는 모든 도면에 대해 한 파일로 한꺼번에 출력함

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            gt_to_dt_match_dict (dict): gt의 심볼 index를 key로, 매칭된 dt의 심볼 index를 value로 갖는 dict
            recognition_result (dict): 도면 이름을 key로, 각 도면에서의 text recognition 계산에 필요한 정보들(recog_num, gt_text_num)을 저장한 dict
            symbol_dict (dict): 심볼 이름을 key로, id를 value로 갖는 dict
        Returns:
            None
        """

        for filename, gt_values in gt_result.items():
            img_path = f"{imgs_path}{filename}.jpg"
            if not os.path.exists(img_path): continue
            print(f"{imgs_path}{filename}.jpg")

            Path(f"{self.output_dir}/tp_cropped/{filename}").mkdir(parents=True, exist_ok=True)
            Path(f"{self.output_dir}/dt_only_cropped/{filename}").mkdir(parents=True, exist_ok=True)

            diagram_img = cv.imread(img_path)

            for gt_index, gt_value in enumerate(gt_values):
                dt_value = {}
                if gt_index in gt_to_dt_match_dict[filename]:
                    dt_value = dt_result[filename][gt_to_dt_match_dict[filename][gt_index]]
                
                if 'category_id' not in gt_value: continue
                if 'category_id' not in dt_value: continue
                if gt_value['category_id'] != (symbol_dict['text'] or symbol_dict['text']): continue
                if dt_value['category_id'] != (symbol_dict['text'] or symbol_dict['text']): continue

                if gt_value['text'] == '': continue 
                if 'bbox' not in dt_value: dt_value['bbox'] = [-1, -1, -1, -1]
                if 'text' not in dt_value: dt_value['text'] = '검출 실패'
                if dt_value['bbox'] == []: dt_value['bbox'] = [-1, -1, -1, -1]
                if dt_value['text'] == '': dt_value['text'] = '텍스트 인식 실패'

                if(gt_value['text'] == dt_value['text']): # recog 성공
                    dt_pos = dt_value['bbox']
                    cropped = diagram_img[dt_pos[1]:dt_pos[1]+dt_pos[3], dt_pos[0]:dt_pos[0]+dt_pos[2]]
                    jpg_name = str.replace(dt_value['text'], '/', '$')
                    write_path = f"{self.output_dir}/tp_cropped/{filename}/{jpg_name}.jpg"
                    idx = 1
                    while True:
                        if not os.path.exists(write_path): break
                        write_path = f"{self.output_dir}/tp_cropped/{filename}/{jpg_name}({idx}).jpg"
                        idx += 1
                    cv.imwrite(write_path, cropped)
                else: # dt 성공, recog 실패
                    dt_pos = dt_value['bbox']
                    cropped = diagram_img[dt_pos[1]:dt_pos[1]+dt_pos[3], dt_pos[0]:dt_pos[0]+dt_pos[2]]
                    jpg_name = str.replace(dt_value['text'], '/', '$')
                    write_path = f"{self.output_dir}/dt_only_cropped/{filename}/{jpg_name}.jpg"
                    idx = 1
                    while True:
                        if not os.path.exists(write_path): break
                        write_path = f"{self.output_dir}/dt_only_cropped/{filename}/{jpg_name}({idx}).jpg"
                        idx += 1
                    cv.imwrite(write_path, cropped)

    def visualize_recog_image(self, gt_result, dt_result, gt_to_dt_match_dict, symbol_dict, imgs_path):
        """ Recognition 결과를 파일로 출력. test 내에 존재하는 모든 도면에 대해 한 파일로 한꺼번에 출력함

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            gt_to_dt_match_dict (dict): gt의 심볼 index를 key로, 매칭된 dt의 심볼 index를 value로 갖는 dict
            recognition_result (dict): 도면 이름을 key로, 각 도면에서의 text recognition 계산에 필요한 정보들(recog_num, gt_text_num)을 저장한 dict
            symbol_dict (dict): 심볼 이름을 key로, id를 value로 갖는 dict
        Returns:
            None
        """

        for filename, gt_values in gt_result.items():
            img_path = f"{imgs_path}{filename}.jpg"
            if not os.path.exists(img_path): continue
            print(f"{imgs_path}{filename}.jpg")

            Path(f"{self.output_dir}/visualized/{filename}").mkdir(parents=True, exist_ok=True)
            Path(f"{self.output_dir}/dt_only_cropped/{filename}").mkdir(parents=True, exist_ok=True)

            diagram_img = cv.imread(img_path)

            for gt_index, gt_value in enumerate(gt_values):
                dt_value = {}
                if gt_index in gt_to_dt_match_dict[filename]:
                    dt_value = dt_result[filename][gt_to_dt_match_dict[filename][gt_index]]
                
                if 'category_id' not in gt_value: continue
                if 'category_id' not in dt_value: continue
                if gt_value['category_id'] != (symbol_dict['text'] or symbol_dict['text']): continue
                if dt_value['category_id'] != (symbol_dict['text'] or symbol_dict['text']): continue

                if gt_value['text'] == '': continue 
                if 'bbox' not in dt_value: dt_value['bbox'] = [-1, -1, -1, -1]
                if 'text' not in dt_value: dt_value['text'] = '검출 실패'
                if dt_value['bbox'] == []: dt_value['bbox'] = [-1, -1, -1, -1]
                if dt_value['text'] == '': dt_value['text'] = '텍스트 인식 실패'

                if(gt_value['text'] == dt_value['text']): # recog 성공
                    dt_pos = dt_value['bbox']
                    cropped = diagram_img[dt_pos[1]:dt_pos[1]+dt_pos[3], dt_pos[0]:dt_pos[0]+dt_pos[2]]
                    jpg_name = str.replace(dt_value['text'], '/', '$')
                    write_path = f"{self.output_dir}/tp_cropped/{filename}/{jpg_name}.jpg"
                    idx = 1
                    while True:
                        if not os.path.exists(write_path): break
                        write_path = f"{self.output_dir}/tp_cropped/{filename}/{jpg_name}({idx}).jpg"
                        idx += 1
                    cv.imwrite(write_path, cropped)
                else: # dt 성공, recog 실패
                    dt_pos = dt_value['bbox']
                    cropped = diagram_img[dt_pos[1]:dt_pos[1]+dt_pos[3], dt_pos[0]:dt_pos[0]+dt_pos[2]]
                    jpg_name = str.replace(dt_value['text'], '/', '$')
                    write_path = f"{self.output_dir}/dt_only_cropped/{filename}/{jpg_name}.jpg"
                    idx = 1
                    while True:
                        if not os.path.exists(write_path): break
                        write_path = f"{self.output_dir}/dt_only_cropped/{filename}/{jpg_name}({idx}).jpg"
                        idx += 1
                    cv.imwrite(write_path, cropped)

    def dump_match_recognition_result(self, gt_result, dt_result, gt_to_dt_match_dict, recognition_result, symbol_dict, recognized_only = False, score_type = "gt"):
        """ Recognition 결과를 파일로 출력. test 내에 존재하는 모든 도면에 대해 한 파일로 한꺼번에 출력함

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            gt_to_dt_match_dict (dict): gt의 심볼 index를 key로, 매칭된 dt의 심볼 index를 value로 갖는 dict
            recognition_result (dict): 도면 이름을 key로, 각 도면에서의 text recognition 계산에 필요한 정보들(recog_num, gt_text_num)을 저장한 dict
            symbol_dict (dict): 심볼 이름을 key로, id를 value로 갖는 dict
        Returns:
            None
        """

        outpath = os.path.join(self.output_dir, f"test_text_match_result_{score_type}.txt")
        if recognized_only: outpath = os.path.join(self.output_dir, f"test_text_match_result_recognized_only_{score_type}.txt")

        with open(outpath, 'w') as f:
            mean_recog_ratio = 0
            total_gt_text_num = 0
            total_tp_text_num = 0

            for filename, gt_values in gt_result.items():
                f.write(f"test drawing : {filename}----------------------------------\n")
                recog_values = recognition_result[filename]
                if score_type == "gt":
                    f.write(f"total recognition ratio: {recog_values['recognized_num']} / {recog_values['all_gt_text_num']} = {recog_values['recognition']}\n")
                elif score_type == "tp":
                    ratio = 0
                    if recog_values['all_tp_text_num'] != 0:
                        ratio = recog_values['recognized_num'] / recog_values['all_tp_text_num']
                    f.write(f"total recognition ratio: {recog_values['recognized_num']} / {recog_values['all_tp_text_num']} = {ratio}\n")
                    ratio = 0

                for gt_index, gt_value in enumerate(gt_values):
                    dt_value = {}
                    if gt_index in gt_to_dt_match_dict[filename]:
                       dt_value = dt_result[filename][gt_to_dt_match_dict[filename][gt_index]]
                    
                    if gt_value['text'] == '': continue 
                    if 'bbox' not in dt_value: dt_value['bbox'] = [-1, -1, -1, -1]
                    if 'text' not in dt_value: dt_value['text'] = '검출 실패'
                    if dt_value['bbox'] == []: dt_value['bbox'] = [-1, -1, -1, -1]
                    if dt_value['text'] == '': dt_value['text'] = '텍스트 인식 실패'

                    cls = ''
                    if gt_value['category_id'] == symbol_dict['text']: cls = 'text'
                    elif gt_value['category_id'] == symbol_dict['text_rotated']: cls = 'text'
                    if(gt_value['text'] == dt_value['text'] or not recognized_only):
                        f.write(f"{cls}| Detected: {dt_value['bbox']}, '{dt_value['text']}', GT: {gt_value['bbox']}, '{gt_value['text']}'\n")

                mean_recog_ratio += recog_values['recognized_num']
                total_gt_text_num += recog_values['all_gt_text_num']
                total_tp_text_num += recog_values['all_tp_text_num']
                f.write("\n")

            if score_type == 'gt': 
                mean_recog_ratio = mean_recog_ratio / total_gt_text_num if total_gt_text_num != 0 else 0
            else:
                mean_recog_ratio = mean_recog_ratio / total_tp_text_num if total_tp_text_num != 0 else 0
            f.write(f"(mean recognition ratio) = ({mean_recog_ratio})")

    def dump_pr_and_ap_result(self, pr_result, ap_result_str, recognition_result, symbol_dict, ap_result_only_sym_str=None, score_type = 'gt'):
        """ AP와 PR 계산 결과를 파일로 출력. test 내에 존재하는 모든 도면에 대해 한 파일로 한꺼번에 출력함

        Arguments:
            pr_result (dict): 도면 이름을 key로, 각 도면에서의 PR 계산에 필요한 정보들(detected_num, gt_num 및 클래스별 gt/dt num)을 저장한 dict
            ap_result_str (string): cocoeval의 evaluate summary를 저장한 문자열
            recognition_result (dict): 도면 이름을 key로, 각 도면에서의 text recognition 계산에 필요한 정보들(recog_num, gt_text_num)을 저장한 dict
            symbol_dict (dict): 심볼 이름을 key로, id를 value로 갖는 dict
            ap_result_only_sym_str (string)_: text class를 제외하고 계산된 cocoeval의 evaluate summary를 저장한 문자열, None인 경우는 text class가 추가되지 않았다고 생각함
        Returns:
            None
        """
        write_only_sym_reslt = ap_result_only_sym_str is not None
        outpath = os.path.join(self.output_dir, f"test_result_{score_type}.txt")
        with open(outpath, 'w') as f:
            mean_precision = 0
            mean_recall = 0
            mean_recog_ratio = 0
            total_gt_text_num = 0
            total_tp_text_num = 0

            if write_only_sym_reslt:
                text_classes_list = []
                if "text" in symbol_dict.keys():
                    text_classes_list.append(symbol_dict["text"])
                if "text_rotated" in symbol_dict.keys():
                    text_classes_list.append(symbol_dict["text_rotated"])
                if "text_rotated_45" in symbol_dict.keys():
                    text_classes_list.append(symbol_dict["text_rotated_45"])
                mean_precision_only_sym = 0
                mean_recall_only_sym = 0

            for filename, values in pr_result.items():
                f.write(f"test drawing : {filename}----------------------------------\n")
                f.write(f"total precision : {values['detected_num']} / {values['all_prediction_num']} = {values['precision']}\n")
                f.write(f"total recall : {values['detected_num']} / {values['all_gt_num']} = {values['recall']}\n")

                recog_values = recognition_result[filename]
                if score_type == 'gt':
                    f.write(f"total recognition ratio: {recog_values['recognized_num']} / {recog_values['all_gt_text_num']} = {recog_values['recognition']}\n")
                elif score_type == 'tp':
                    ratio = 0
                    if recog_values['all_tp_text_num'] != 0:
                        ratio = recog_values['recognized_num'] / recog_values['all_tp_text_num']
                    f.write(f"total recognition ratio: {recog_values['recognized_num']} / {recog_values['all_tp_text_num']} = {ratio}\n")
                    ratio = 0

                mean_precision += values['precision']
                mean_recall += values['recall']
                mean_recog_ratio += recog_values['recognized_num']
                total_gt_text_num += recog_values['all_gt_text_num']
                total_tp_text_num += recog_values['all_tp_text_num']

                if write_only_sym_reslt:
                    only_sym_detected_num = values['detected_num']
                    only_sym_all_prediction_num = values['all_prediction_num']
                    only_sym_all_gt_num = values['all_gt_num']

                    only_text_detected_num = 0
                    only_text_prediction_num = 0
                    only_text_gt_num = 0
                    for gt_class, gt_num, detected_num, detection_num in zip(values["gt_classes"], values["per_class_gt_num"],values["per_class_detected_num"], values['per_class_detection_num']):
                        if gt_class in text_classes_list:
                            only_sym_detected_num -= detected_num
                            only_sym_all_prediction_num -= detection_num
                            only_sym_all_gt_num -= gt_num

                            only_text_gt_num += gt_num
                            only_text_detected_num += detected_num
                            only_text_prediction_num += detection_num

                    f.write(f"\nonly symbol precision : {only_sym_detected_num} / {only_sym_all_prediction_num} = {only_sym_detected_num/only_sym_all_prediction_num}\n")
                    f.write(f"only symbol recall : {only_sym_detected_num} / {only_sym_all_gt_num} = {only_sym_detected_num/only_sym_all_gt_num}\n")

                    mean_precision_only_sym += only_sym_detected_num/only_sym_all_prediction_num
                    mean_recall_only_sym += only_sym_detected_num/only_sym_all_gt_num
                    
                for gt_class, gt_num, detected_num in zip(values["gt_classes"], values["per_class_gt_num"],values["per_class_detected_num"]):
                    if symbol_dict is not None:
                        sym_name = [k for k,v in symbol_dict.items() if v == gt_class]
                    else:
                        sym_name = ""
                    f.write(f"class {gt_class} ({sym_name}) : {detected_num} / {gt_num}\n")

                f.write("\n")
            f.write(ap_result_str)

            if write_only_sym_reslt:
                mean_precision_only_sym /= len(pr_result.keys())
                mean_recall_only_sym /= len(pr_result.keys())

                ap_strs = ap_result_only_sym_str.splitlines()[0].split(" ")
                ap = float(ap_strs[len(ap_strs) - 1])
                ap_50_strs = ap_result_only_sym_str.splitlines()[1].split(" ")
                ap_50 = float(ap_50_strs[len(ap_50_strs) - 1])
                ap_75_strs = ap_result_only_sym_str.splitlines()[2].split(" ")
                ap_75 = float(ap_75_strs[len(ap_75_strs) - 1])
                f.write(f"(mean precision only sym, mean recall only sym, ap, ap50, ap75) = ({mean_precision_only_sym}, {mean_recall_only_sym}, {ap}, {ap_50}, {ap_75})\n")

            mean_precision /= len(pr_result.keys())
            mean_recall /= len(pr_result.keys())
            if score_type == 'gt': mean_recog_ratio = mean_recog_ratio / total_gt_text_num if total_gt_text_num != 0 else 0
            else: mean_recog_ratio = mean_recog_ratio / total_tp_text_num if total_tp_text_num != 0 else 0

            ap_strs = ap_result_str.splitlines()[0].split(" ")
            ap = float(ap_strs[len(ap_strs)-1])
            ap_50_strs = ap_result_str.splitlines()[1].split(" ")
            ap_50 = float(ap_50_strs[len(ap_50_strs) - 1])
            ap_75_strs = ap_result_str.splitlines()[2].split(" ")
            ap_75 = float(ap_75_strs[len(ap_75_strs) - 1])

            f.write(f"(mean precision, mean recall, mean recognition ratio, ap, ap50, ap75) = ({mean_precision}, {mean_recall}, {mean_recog_ratio}, {ap}, {ap_50}, {ap_75})")

    def calculate_ap(self, gt_result_json, dt_result, ignore_class_list=None):
        """ COCOeval을 사용한 AP계산. 중간 과정으로 gt와 dt에 대한 json파일이 out_dir에 생성됨

        Arguments:
            gt_result_json (dict): test 내에 존재하는 모든 도면에 대한 images, annotation, category 정보를 coco json 형태로 저장한 dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            ignore_class_list (list[int]) : cocoeval 계산에서 무시할 class의 리스트 예를들어 [12, 35, 68] 이라면 이 세 class 제외하고 계산
        Returns:
            result_str (string): COCOeval의 계산 결과 summary 저장한 문자열
        """
        # 먼저 gt_json을 파일로 출력
        gt_outpath = os.path.join(self.output_dir, "test_gt_global.json")
        coco_json_write(gt_outpath, gt_result_json)

        # dt_result를 coco형식으로 변환하여 파일로 출력 (주의! dt는 NMS 이전의 결과여야 함)
        test_dt_global = []
        for filename, bboxes in dt_result.items():
            for box in bboxes:
                box["image_id"] = self.get_gt_img_id_from_filename(filename, gt_result_json)
                test_dt_global.append(box)

        dt_outpath = os.path.join(self.output_dir, "test_dt_global.json")
        coco_json_write(dt_outpath, test_dt_global)

        # gt와 dt파일을 로드하여 ap 계산
        cocoGT = coco.COCO(gt_outpath)
        cocoDt = cocoGT.loadRes(dt_outpath)
        annType = 'bbox'
        cocoEval = cocoeval.COCOeval(cocoGT,cocoDt,annType)

        # ignore class 제거
        if ignore_class_list is not None:
            for ignore_class in ignore_class_list:
                cocoEval.params.catIds.remove(ignore_class)

        cocoEval.evaluate()
        cocoEval.accumulate()

        original_stdout = sys.stdout
        string_stdout = io.StringIO()
        sys.stdout = string_stdout
        cocoEval.summarize()
        sys.stdout = original_stdout

        result_str = string_stdout.getvalue()

        return result_str

    def calculate_pr(self, gt_result, dt_result, gt_to_dt_match_dict):
        """ 전체 test 도면에 대한 precision 및 recall 계산

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            gt_to_dt_match_dict (dict): gt 심볼과 dt 심볼간 매칭 계산 결과 dict

        Returns:
            pr_result (dict): 도면 이름을 key로, Precision 및 recall 계산에 필요한 데이터를 value로 갖는 dict
        """
        pr_result = {}
        for filename, bboxes in dt_result.items():

            dt_class_num_dict = defaultdict(int)
            gt_class_num_dict = defaultdict(int)
            gt_detected_class_num_dict = defaultdict(int)

            detected_num = 0
            all_prediction = len(bboxes)

            for box in bboxes:
                dt_class_num_dict[box['category_id']] += 1

            for gt_annotation in gt_result[filename]:
                gt_class_num_dict[gt_annotation["category_id"]] += 1

            gt_num = len(gt_result[filename])

            for gt_index in gt_to_dt_match_dict[filename].keys():
                gt_bbox = gt_result[filename][gt_index]
                gt_class = gt_bbox["category_id"]
                gt_detected_class_num_dict[gt_class] += 1
                detected_num += 1

            gt_classes = sorted(list(gt_class_num_dict.keys()))

            per_class_gt_num = []
            per_class_detection_num = []
            per_class_detected_num = []
            for class_index in gt_classes:
                per_class_gt_num.append(gt_class_num_dict[class_index])
                per_class_detected_num.append(gt_detected_class_num_dict[class_index])
                per_class_detection_num.append(dt_class_num_dict[class_index])

            pr_result[filename] = { "all_gt_num" : gt_num,
                                    "all_prediction_num" : all_prediction,
                                    "detected_num" : detected_num,
                                    "gt_classes" : gt_classes,
                                    "per_class_detection_num": per_class_detection_num,
                                    "per_class_detected_num" : per_class_detected_num,
                                    "per_class_gt_num" : per_class_gt_num,
                                    "precision" : detected_num/all_prediction,
                                    "recall" : detected_num/gt_num
                                    }

        return pr_result
    
    def calculate_recognition(self, gt_result, dt_result, gt_to_dt_match_dict):
        """ 전체 test 도면에 대한 recognition 계산

        Arguments:
            gt_result (dict): 도면 이름을 key로, box들을 value로 갖는 gt dict
            dt_result (dict): 도면 이름을 key로, box들을 value로 갖는 dt dict
            gt_to_dt_match_dict (dict): gt 심볼과 dt 심볼간 매칭 계산 결과 dict

        Returns:
            pr_result (dict): 도면 이름을 key로, recognition 계산에 필요한 데이터를 value로 갖는 dict
        """

        recognition_result = {}

        for filename, gt_to_dt_match in gt_to_dt_match_dict.items():
            recog_count = 0
            gt_text_count = 0
            tp_text_count = 0
            for gt_id, dt_id in gt_to_dt_match.items():
                gt_category = gt_result[filename][gt_id]['category_id']
                if gt_category != 499 and gt_category != 500: continue
                dt_category = dt_result[filename][dt_id]['category_id']
                if dt_category != 499 and dt_category != 500: continue
                if 'text' not in dt_result[filename][dt_id]: continue
                tp_text_count += 1
                if gt_result[filename][gt_id]['text'] == dt_result[filename][dt_id]['text']: 
                    recog_count += 1

            for gt_value in gt_result[filename]:
                if gt_value['category_id'] == 499 or gt_value['category_id'] == 500: gt_text_count += 1 
            
            recognition_result[filename] = {"all_gt_text_num" : gt_text_count, "all_tp_text_num": tp_text_count, "recognized_num" : recog_count, "recognition" : recog_count / gt_text_count}
        return recognition_result

    def get_gt_img_id_from_filename(self, filename, gt_result_json):
        """ filename을 입력으로 img_id를 반환하는 함수

        Arguments:
            filename (string): 파일이름 (ex, "KNU-A-36420-014.jpg")
            gt_result_json (dict): test 내에 존재하는 모든 도면에 대한 images, annotation, category 정보를 coco json 형태로 저장한 dict
        Return:
            gt_result_json에 기록된 filename에 해당하는 도면의 id
        """
        for imgs in gt_result_json['images']:
            if filename == imgs["file_name"].split(".")[0]:
                return imgs["id"]