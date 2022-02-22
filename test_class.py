from utils_class import Init_Config
from det_inference_class import Detection_Inference
from landmark_inference_class import Landmark_Inference
from facenet_inference_class import Facenet_Inference
import lt_sdk as lt
import pdb
from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
from lt_sdk.graph.transform_graph import utils as lt_utils
import cv2
# from pfld_landmark_infer import pfld_infer_process, read_img_input, Coordinate_landmarks_map, align_face
from common import parse_csv_get_feas_list, face_recognition
from utils.plot_utils import plot_one_box
config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)

if __name__ == '__main__':
    yolov3_lt_graph_path = './lt_graph_pb/yolov3_tiny_lgf_end.pb' 
    facenet_lt_graph_path = './lt_graph_pb/facenet_lgf_end.pb'
    pfld_lt_graph_path = './lt_graph_pb/pfld_lgf_end2.pb' 

    input_image = 'test.jpg'
    features_csv_path = 'face_features_lib.csv'
    anchor_path = 'data/yolov3_tiny_widerface_anchors.txt'
    class_name_path = 'data/widerface.names'
    init_cfg = Init_Config(class_name_path, anchor_path, features_csv_path)
    img_ori, anchors, classes, color_table, features_csv_list = init_cfg.init_config(input_image)

    det_model = Detection_Inference("input_data", 224, classes, 1, anchors, yolov3_lt_graph_path, config)
    landmark_model = Landmark_Inference("input", pfld_lt_graph_path, 1, 112, 112, config)
    facenet_model = Facenet_Inference("image_batch", facenet_lt_graph_path, 160, 160, 1, config)
    boxes, scores, labels = det_model.run(input_image)
    print("boxes = ", boxes)
    print("scores = ", scores)
    print("labels = ", labels)

    face_features = []
    face_landmarks = []
    names = []
    for i in range(len(boxes)):
        x0, y0, x1, y1 = boxes[i]
        # x0, y0, x1, y1 = x0-30, y0-30, x1+30, y1+30
        # boxes_list = [x0, y0, x1, y1]
        # boxes_npy = np.array(boxes_list)
        face_crop = img_ori[int(y0):int(y1), int(x0):int(x1)]
        ########### face landmark detection start ###########
        rotated_img, landmark_map_res, theatas = landmark_model.run(face_crop, img_ori, boxes[i])
        # ########### face landmark detection end #############
        face_landmarks.append(landmark_map_res)
        
        features = facenet_model.run(rotated_img)
        fea_map = features[0].tolist()[0]
        face_features.append(fea_map)

        name = face_recognition(fea_map, features_csv_list)
        if name == None:
            name = 'Unkown'
        print("name = ", name)
        names.append(name)
        
        ############ draw landmark ######################
        for p in range(int(landmark_map_res.shape[1]/2)):
            cv2.circle(img_ori, (int(landmark_map_res[0][p*2+0]), int(landmark_map_res[0][p*2+1])), 2, (0, 0, 255), -1)
        ############ draw landmark end ##################
        plot_one_box(img_ori, [x0, y0, x1, y1], label=name+' '+classes[labels[i]], color=color_table[labels[i]])
    cv2.imwrite('detection_result.jpg', img_ori)
    # return boxes, scores, labels, face_features, face_landmarks, names
