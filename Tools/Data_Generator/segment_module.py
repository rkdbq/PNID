import numpy as np

def segment_symbols(objects, drawing_resize_scale):
    bbox_array = np.zeros((len(objects),4))
    for ind in range(len(objects)):
        objects[ind] = [objects[ind][0],
                        int(objects[ind][1]*drawing_resize_scale),
                        int(objects[ind][2]*drawing_resize_scale),
                        int(objects[ind][3]*drawing_resize_scale),
                        int(objects[ind][4]*drawing_resize_scale)]
        bbox_object = objects[ind]
        bbox_array[ind, :] = np.array([bbox_object[1] , bbox_object[2], bbox_object[3], bbox_object[4]])
    return bbox_array

def segment_text(txt_object_list, drawing_resize_scale):
  txt_bbox_array = np.zeros((len(txt_object_list), 4))
  for ind in range(len(txt_object_list)):
      txt_object_list[ind] = [txt_object_list[ind][0],
                              int(txt_object_list[ind][1] * drawing_resize_scale),
                              int(txt_object_list[ind][2] * drawing_resize_scale),
                              int(txt_object_list[ind][3] * drawing_resize_scale),
                              int(txt_object_list[ind][4] * drawing_resize_scale),
                              int(txt_object_list[ind][5])]
      txt_bbox_object = txt_object_list[ind]
      txt_bbox_array[ind, :] = np.array([txt_bbox_object[1], txt_bbox_object[2], txt_bbox_object[3], txt_bbox_object[4]])
  return txt_bbox_array

def index_objects(bbox_array, start_width, start_height, width_size, height_size):
  xmin_in = bbox_array[:, 0] > start_width
  ymin_in = bbox_array[:, 1] > start_height
  xmax_in = bbox_array[:, 2] < start_width+width_size
  ymax_in = bbox_array[:, 3] < start_height+height_size
  is_bbox_in = xmin_in & ymin_in & xmax_in & ymax_in
  in_bbox_ind = [i for i, val in enumerate(is_bbox_in) if val == True]
  return in_bbox_ind

def segment_image(img, start_width, start_height, width_size, height_size):
  sub_img = np.ones((height_size, width_size, 3)) * 255
  if start_width+width_size > img.shape[1] and start_height+height_size > img.shape[0]:
      sub_img[0:img.shape[0] - (start_height + height_size),
              0:img.shape[1] - (start_width + width_size), :] = img[start_height:start_height + height_size,
                                                                    start_width:start_width + width_size, :]
  elif start_width+width_size > img.shape[1]:
      sub_img[:, 0:img.shape[1]-(start_width+width_size), :] = img[start_height:start_height+height_size, start_width:img.shape[1],:]
  elif start_height+height_size > img.shape[0]:
      sub_img[0:img.shape[0] - (start_height + height_size), : , :] = img[start_height:img.shape[0], start_width:start_width+width_size, :]
  else:
      sub_img = img[start_height:start_height+height_size, start_width:start_width+width_size, :]
  return sub_img