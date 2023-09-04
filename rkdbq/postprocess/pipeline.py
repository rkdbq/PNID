import datetime
from pathlib import Path
from post_process import diagram_text_to_dic, convert_class_to_diagram, calculate_rotated_pr, symbol_dict_text_to_dic, dump_rotated_pr_result
from merge import text_merge
from tqdm import tqdm

exp_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

######################################################################

# class annfile to diagram annfile

## paths
cls_anns_path = ""
dt_anns_path = "" # to save diagram annfile

### perform
convert_class_to_diagram(cls_anns_path, dt_anns_path)

######################################################################

# evaluate precision & recall

## paths
gt_anns_path = ""
dt_anns_path = ""
symbol_dict_path = ""
eval_result_txt_path = "" # to save evaluated result txt file

### perform
gt_result = diagram_text_to_dic(gt_anns_path)
dt_result = diagram_text_to_dic(dt_anns_path)
pr_result = calculate_rotated_pr(gt_result, dt_result)
symbol_dict = symbol_dict_text_to_dic(symbol_dict_path)
dump_rotated_pr_result(pr_result, symbol_dict)

######################################################################

# merge duplicated text bounding box

## paths
dt_anns_path = ""
merged_dt_anns_path = "" # to save merged DT (diagram) annfile

### perform
merge = text_merge(dt_anns_path).write(merged_dt_anns_path)

######################################################################

# visualize on GT diagram

## paths
gt_imgs_path = ""
gt_anns_path = ""
dt_anns_path = ""
vis_imgs_path = "" # to save visualized image

### perform
gt_result = diagram_text_to_dic(gt_anns_path)
dt_result = diagram_text_to_dic(dt_anns_path)

# 함수화 필요
# for diagram in tqdm(gt_result.keys(), desc="Text Visualization"):
#     diagram_path = f"{images_dir}{diagram}.jpg"
#     save_path = f"{vis_dir}{diagram}.jpg"
#     draw_rectangle_and_save(diagram_path, save_path, gt_result[diagram], dt_result[diagram])
