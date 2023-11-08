from Predict_Postprocess.gt_dt_data import gt_dt_data
from Predict_Postprocess.evaluate import evaluate
from Common.symbol_io import read_symbol_type_txt
from pathlib import Path
import shutil

# Test 결과의 성능 계산 및 이미지 출력 코드

gt_json_filepath = "D:\\Data\\json2eval\\GT_json\\test.json"  # 학습 도면 분할시 생성한 test.json 파일 경로
dt_json_filepath = "D:\\Data\\json2eval\\DT_json\\gt_text_pred_latest.json"  # prediction 결과로 mmdetection에서 생성된 json 파일 경로
output_dir = "D:\\Experiments\\Recognitions\\gt_text_pred_latest_2\\"  # 출력 파일들이 저장될 폴더

data_root = 'D:\\Data\\raw\\PNID_RAW\\'
drawing_dir = data_root + "Drawing\\JPG\\"  # 원본 도면 이미지 폴더
symbol_xml_dir = data_root +"SymbolXML"  # 원본 도면 이미지와 함께 제공된 Symbol XML 폴더
text_xml_dir = data_root +"TextXML"  # 원본 도면 이미지와 함께 제공된 Text XML 폴더
symbol_filepath = data_root +"Hyundai_SymbolClass_Sym_Only.txt"  # (방향 제거된) symbol index txt 파일 경로
symbol_type_filepath = data_root +"Hyundai_SymbolClass_Type.txt"  # 심볼이름-타입 매칭 txt

include_text_as_class = True
include_text_orientation_as_class = False

stride_w = 300  # 학습 도면 분할시에 사용한 stride
stride_h = 300
drawing_resize_scale = 0.5 # 학습 도면 분할시에 사용한 scaling factor (절반 크기로 줄였으면 0.5)

score_threshold = 0.5  # score filtering threshold
nms_threshold = 0.0
matching_iou_threshold = 0.5  # 매칭(정답) 처리할 IOU threshold
adaptive_thr_dict = {
    311: 0.02, 66: 0.06, 145: 0.4,
    131: 0.65, 431: 0.04, 109: 0.03,
    239: 0.3, 12: 0.0002, 499: 0.2
}

vertical_threshold = 2 # 세로 문자열로 판단하는 기준. 세로가 가로보다 vertical_threshold 배 이상 길면 세로로 판단
text_img_margin_ratio = 0.1  # detection된 문자열에서 크기를 약간 키워서 text recognition을 수행할 경우. (ex, 0.1이면 box를 1.1배 키워서 인식)

# 0) 출력 파일이 저장될 디렉터리가 없다면 자동으로 생성, 사용한 parameter 확인을 위해 이 python script 같이 저장
Path(output_dir).mkdir(parents=True, exist_ok=True)
shutil.copy(__file__, output_dir)

# 1) gt_dt_data 클래스 초기화를 통한 데이터 전처리
#   : 분할 Ground Truth(gt) 및 detection(dt) result를 기반으로 분할 전 도면 좌표로 다시 맵핑하고, score filtering, NMS를 수행
gt_dt_result = gt_dt_data(gt_json_filepath, dt_json_filepath, drawing_dir, symbol_xml_dir, symbol_filepath,
                          include_text_as_class, include_text_orientation_as_class, text_xml_dir,
                          drawing_resize_scale, stride_w, stride_h,
                          score_threshold, nms_threshold, adaptive_thr_dict=adaptive_thr_dict)

symbol_dict = gt_dt_result.symbol_dict  # or read_symbol_txt(symbol_filepath)
symbol_type_dict = read_symbol_type_txt(symbol_type_filepath)

# 2) evaluate 클래스 초기화 및 매칭 정보 생성
#   : NMS 완료된 dt result와 gt result간의 매칭 dictionary 생성
eval = evaluate(output_dir)
gt_to_dt_match_dict, dt_to_gt_match_dict = eval.compare_gt_and_dt(gt_dt_result.gt_result,
                                                                  gt_dt_result.dt_result_after_nms,
                                                                  matching_iou_threshold)

# 3) precision-recall 성능 및 AP 성능 계산 및 Dump
#   : 위에서 얻은 정보들을 바탕으로 성능 계산(주의! AP 계산에는 NMS 하기 전의 결과가 전달되어야 함 (gt_dt_result.dt_result))
pr_result = eval.calculate_pr(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict)
ap_result_str = eval.calculate_ap(gt_dt_result.gt_result_json, gt_dt_result.dt_result)
recog_result = eval.calculate_recognition(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict)

score_type = 'gt'
eval.dump_pr_and_ap_result(pr_result, ap_result_str, recog_result, gt_dt_result.symbol_dict, score_type=score_type)
eval.dump_match_recognition_result(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict, recog_result, gt_dt_result.symbol_dict, score_type=score_type)
eval.dump_match_recognition_result(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict, recog_result, gt_dt_result.symbol_dict, recognized_only=True, score_type=score_type)

score_type = 'tp'
eval.dump_pr_and_ap_result(pr_result, ap_result_str, recog_result, gt_dt_result.symbol_dict, score_type=score_type)
eval.dump_match_recognition_result(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict, recog_result, gt_dt_result.symbol_dict, score_type=score_type)
eval.dump_match_recognition_result(gt_dt_result.gt_result, gt_dt_result.dt_result_after_nms, gt_to_dt_match_dict, recog_result, gt_dt_result.symbol_dict, recognized_only=True, score_type=score_type)