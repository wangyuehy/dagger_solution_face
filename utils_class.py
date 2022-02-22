from utils.misc_utils import parse_anchors, read_class_names
from utils.plot_utils import get_color_table, plot_one_box
from common import parse_csv_get_feas_list
import cv2
class Init_Config(object):
    def __init__(self, class_name_path, anchor_path, features_csv_path):
        self.class_name_path = class_name_path
        self.anchor_path = anchor_path
        self.features_csv_path = features_csv_path

    def init_config(self, input_image):
        img_ori = cv2.imread(input_image)
        anchors = parse_anchors(self.anchor_path)
        classes = read_class_names(self.class_name_path)
        num_class = len(classes)
        color_table = get_color_table(num_class)
        features_csv_list = parse_csv_get_feas_list(self.features_csv_path)
        return img_ori, anchors, classes, color_table, features_csv_list

    
        