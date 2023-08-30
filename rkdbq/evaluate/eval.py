# 각 도면(xml)의 precision, recall 계산
# 전체 도면(xml)에 대한 precision, recall 계산
# 계산 시 모든 심볼/작은 심볼/큰 심볼 선택 가능하여야 함
# 계산 결과를 텍스트로 출력 

def xml2dict(xml_path: str):
    return

def cal_iou(gt_points: list, dt_points: list):
    return

def evaluate(gt_dict: dict, dt_dict: dict):
    precision = {}
    recall = {}

    # for diagram in gt_dict.keys():
    #     if diagram not in dt_dict: 
    #         print(f'{diagram} is skipped. (NOT exist in detection xmls path)\n')
    #         continue
    #     for gt_bboxs, dt_bboxs in zip(gt_dict[diagram], dt_dict[diagram]):
            
    return precision, recall

def dump(precision: dict, recall: dict, recognition: dict = {}, symbol: str = 'total'):
    """
    symbol(text) is 'total', 'small' or 'large'

    precision(dict): {도면 이름: {클래스 이름: {tp, dt}}, ... total: {tp, dt}}}, ..., mean: sum(tp) / sum(dt)}
    recall(dict): {도면 이름: {클래스 이름: {tp, gt}}, ... total: {tp, gt}}}, ..., mean: sum(tp) / sum(gt)}
    recognition(dict): {도면 이름: {tp, dt}, ..., mean: {sum(tp), sum(dt)}}
    """
    return

# pipeline

iou_thr = 0

gt_xmls = ""
dt_xmls = ""

gt_dict = xml2dict(gt_xmls)
dt_dict = xml2dict(dt_xmls)

precision, recall = evaluate(gt_dict, dt_dict)

dump(precision, recall)