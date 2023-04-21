import os
import cv2
from Misc.rotated_box import rotated_box
from post_process import diagram_text_to_dic

# XML 데이터 검증을 위한 가시화 코드

drawing_img_dir = r"//Users//rkdbg//Codes//GitHub//tuna1210//PNID//PNID_DOTA_before_split//test//images//"
output_img_dir = r"//Users//rkdbg//Codes//GitHub//tuna1210//PNID//visualized_images//"
diagram_dir = r"//Users//rkdbg//Codes//GitHub//tuna1210//PNID//PNID_DOTA_before_split//test//detected//"

dt_result = diagram_text_to_dic(diagram_dir)

for diagram in dt_result.keys():
    drawing_path = os.path.join(drawing_img_dir, diagram + ".jpg")
    object_list = dt_result[diagram]

    img = cv2.imread(drawing_path, cv2.IMREAD_COLOR)
    
    for obj in object_list:
        rotated_box(vertices=obj[0:8], text=obj[8]).draw(img)

    out_path = os.path.join(output_img_dir, diagram + ".jpg")

    cv2.imwrite(out_path, img)
    print(f"Visualizing {diagram}...")

print("Visualizing Done")