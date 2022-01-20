import os
import sys
import numpy as np
from scipy.spatial.distance import cosine
import re

def numpy_extractor_info(input_numpy):
    input_list = input_numpy.tolist()[0]
    return input_list

def parse_csv_get_feas_list(features_csv_path):
    feas_map_list = []
    with open(features_csv_path) as f:
        for line in f:
            temp_list = []
            info = line.strip().split(',')
            # print("info number = ", type(info))
            for i in range(len(info)):
                if i == 0:
                    temp_list.append(info[i])
                elif i == 1:
                    temp_list.append(float(info[i][2:]))
                elif i == 512:
                    temp_list.append(float(info[i][:-2]))
                else:
                    temp_list.append(float(info[i]))

            feas_map_list.append(temp_list)
    return feas_map_list
            # for i in range(len(info)):
            #     print("info[{}] = {}".format(i, info[i]))

def face_recognition(feas, feas_list, threshold=0.75):
    name_list = []
    similarity_list = []
    print("feas_list len = ", len(feas_list))
    for i in range(len(feas_list)):
        name = feas_list[i][0]
        name_list.append(name)
        print("feas_list[i] type = ", type(feas_list[i]))
        # feas_list[i].pop(0)
        feas_dst = feas_list[i][1:]
        s = cosine_similarity(feas, feas_dst)
        print("s = ", s)
        similarity_list.append(s)
    index = similarity_list.index(max(similarity_list))
    if max(similarity_list) > threshold:
        return name_list[index]
    else:
        return None

def cosine_similarity(vec_src, vec_dst):
    s = cosine(vec_src, vec_dst)
    return 1. - s

if __name__ == '__main__':
    features_csv_path = 'face_features_lib.csv'
    feas_map_list = parse_csv_get_feas_list(features_csv_path)
    fea1 = [0.0096435546875, 0.10693359375, 0.0074462890625, 0.04541015625, -0.08837890625, -0.047607421875, 0.0791015625, -0.0213623046875, -0.03857421875, -0.0081787109375, -0.02490234375, 0.0224609375, 0.0, 0.0166015625, 0.0478515625, -0.0067138671875, -0.060302734375, 0.00982666015625, 0.01251220703125, -0.03759765625, -0.050048828125, 0.034423828125, -0.0283203125, 0.048095703125, 0.0751953125, -0.0030517578125, 0.014892578125, 0.0166015625, 0.0184326171875, -0.06201171875, 0.009521484375, -0.02392578125, 0.044677734375, -0.046875, 0.07470703125, -0.0184326171875, -0.0634765625, -0.0478515625, -0.0030517578125, -0.0079345703125, -0.06591796875, -0.061279296875, -0.01434326171875, -0.00439453125, 0.0625, -0.044677734375, -0.05419921875, 0.0008392333984375, -0.0096435546875, 0.045654296875, 0.001953125, -0.0284423828125, 0.015869140625, 0.049072265625, -0.0234375, -0.01068115234375, -0.017578125, -0.031005859375, -0.052978515625, 0.03857421875, 0.0247802734375, 0.10986328125, -0.031005859375, -0.0228271484375, 0.046142578125, 0.0130615234375, 0.059814453125, 0.060546875, 0.0291748046875, 0.006072998046875, 0.0615234375, -0.024658203125, -0.0299072265625, 0.049560546875, 0.053955078125, -0.06298828125, 0.0206298828125, -0.046630859375, 0.0419921875, -0.007720947265625, 0.0240478515625, 0.03125, -0.0032196044921875, 0.025634765625, -0.06982421875, -0.000823974609375, 0.045654296875, -0.08935546875, 0.041748046875, 0.00421142578125, 0.051513671875, 0.037353515625, 0.052001953125, 0.06396484375, 0.047119140625, 0.01806640625, -0.11328125, 0.0361328125, -0.03369140625, -0.018310546875, 0.06884765625, 0.04638671875, 0.041748046875, -0.0186767578125, -0.0849609375, -0.0115966796875, -0.0673828125, -0.019775390625, -0.0223388671875, -0.07470703125, -0.07568359375, 0.0791015625, 0.053466796875, -0.0517578125, 0.080078125, 0.0498046875, 0.0260009765625, 0.039794921875, 0.01336669921875, -0.054931640625, -0.047607421875, -0.076171875, 0.0947265625, 0.05517578125, 0.031982421875, -0.00823974609375, -0.03955078125, -0.07275390625, 0.0517578125, -0.01165771484375, 0.08203125, 0.02001953125, -0.025146484375, 0.0162353515625, -0.0284423828125, -0.057861328125, -0.0595703125, 0.050537109375, 0.03271484375, 0.0390625, -0.06103515625, -0.037109375, -0.051025390625, 0.024658203125, -0.03271484375, -0.0186767578125, -0.0162353515625, 0.033935546875, 0.0205078125, 0.051025390625, 0.0198974609375, 0.02880859375, 0.005645751953125, 0.01031494140625, 0.0289306640625, -0.078125, 0.025634765625, 0.040283203125, -0.034912109375, -0.06494140625, 0.0084228515625, -0.0517578125, 0.02001953125, -0.0263671875, -0.0228271484375, -0.010986328125, 0.0576171875, -0.0150146484375, -0.00933837890625, -0.078125, 0.0201416015625, 0.0380859375, -0.07080078125, 0.0274658203125, 0.0286865234375, -0.0284423828125, 0.0859375, -0.033203125, 0.0693359375, -0.019775390625, 0.0137939453125, -0.0274658203125, -0.006378173828125, -0.048583984375, 0.0380859375, 0.02392578125, -0.01953125, 0.01116943359375, 0.04052734375, -0.0400390625, -0.06494140625, -0.0234375, -0.03271484375, 0.040283203125, -0.0693359375, -0.00885009765625, 0.0830078125, 0.0228271484375, 0.03173828125, -0.10009765625, 0.080078125, 0.058349609375, -0.031982421875, 0.056884765625, -0.051513671875, 0.01141357421875, -0.016357421875, 0.056640625, 0.0252685546875, -0.15234375, -0.0172119140625, -0.06787109375, 0.038818359375, 0.04443359375, 0.056640625, -0.047607421875, 0.0269775390625, -0.030029296875, 0.00421142578125, 0.04296875, -0.003387451171875, -0.01904296875, 0.032958984375, 0.0220947265625, -0.038818359375, -0.00732421875, -0.029296875, 0.03173828125, -0.08544921875, 0.04638671875, 0.0341796875, 0.056396484375, -0.034912109375, -0.005218505859375, 0.05224609375, -0.003204345703125, 0.033935546875, -0.051025390625, 0.03466796875, -0.031982421875, -0.080078125, -0.036865234375, -0.045166015625, -0.0084228515625, -0.01806640625, 0.00762939453125, 0.0242919921875, -0.039794921875, 0.01708984375, -0.0732421875, -0.07470703125, 0.02490234375, 0.0927734375, 0.057373046875, 0.04345703125, -0.04736328125, 0.0166015625, -0.01190185546875, 0.044921875, 0.0252685546875, -0.0478515625, -0.038818359375, -0.0908203125, 0.01318359375, -0.0634765625, -0.03271484375, 0.01153564453125, 0.003692626953125, -0.054931640625, 0.08837890625, -0.0277099609375, -0.0137939453125, -0.04248046875, 0.0380859375, 0.07763671875, 0.07861328125, 0.03564453125, 0.06494140625, 0.0084228515625, -0.01324462890625, -0.04638671875, 0.0673828125, 0.05029296875, -0.0223388671875, 0.03857421875, -0.002471923828125, -0.0235595703125, 0.0810546875, 0.0299072265625, 0.006591796875, -0.06884765625, 0.04052734375, -0.026123046875, -0.08837890625, -0.039794921875, -0.051025390625, 0.0089111328125, 0.087890625, 0.07861328125, 0.0235595703125, 0.0732421875, 0.06640625, 0.0233154296875, -0.11279296875, 0.00994873046875, 0.03759765625, -0.046875, 0.007537841796875, 0.0238037109375, 0.0233154296875, -0.10791015625, 0.0240478515625, 0.053466796875, 0.0019683837890625, -0.04296875, -0.01348876953125, -0.02392578125, 0.059326171875, -0.0108642578125, -0.0712890625, 0.0361328125, -0.027587890625, -0.04931640625, -0.083984375, 0.007110595703125, 0.016357421875, 0.022705078125, -0.0311279296875, 0.0390625, -0.0067138671875, 0.0272216796875, 0.06884765625, 0.083984375, -0.0147705078125, 0.033935546875, -0.01202392578125, -0.00396728515625, -0.10546875, 0.0673828125, -0.039794921875, 0.07421875, -0.07421875, -0.0546875, 0.03955078125, 0.012939453125, 0.0751953125, 0.021240234375, -0.076171875, -0.031494140625, -0.0036163330078125, 0.09765625, 0.027099609375, -0.0252685546875, 0.055908203125, 0.024658203125, -0.037841796875, -0.007659912109375, -0.028076171875, -0.033935546875, -0.00885009765625, -0.01495361328125, 0.0419921875, 0.00537109375, -0.000965118408203125, -0.05517578125, -0.0849609375, 0.006744384765625, 0.010498046875, -0.04638671875, -0.0732421875, 0.01251220703125, 0.0673828125, -0.037109375, 0.020751953125, 0.080078125, 0.01055908203125, 0.0206298828125, 0.001800537109375, -0.049560546875, 0.04541015625, -0.01483154296875, 0.0030670166015625, 0.00860595703125, -0.06298828125, 0.0181884765625, -0.0576171875, 0.036865234375, 0.0283203125, 0.0211181640625, -0.0255126953125, -0.01123046875, 0.03466796875, 0.002227783203125, 0.025146484375, 0.05419921875, 0.0113525390625, 0.00994873046875, 0.007293701171875, -0.031494140625, -0.08935546875, -0.0791015625, 0.024658203125, -0.0106201171875, 0.00787353515625, 0.06298828125, 0.0255126953125, 0.06298828125, -0.023681640625, -0.0299072265625, 0.038818359375, -0.03564453125, 0.003936767578125, 0.03369140625, -0.076171875, 0.0380859375, -0.031005859375, -0.0142822265625, 0.04541015625, -0.0693359375, -0.032958984375, -0.00274658203125, -0.023193359375, -0.0230712890625, 0.014892578125, -0.10205078125, 0.00921630859375, -0.0311279296875, -0.07958984375, 0.068359375, -0.03271484375, 0.0286865234375, -0.0311279296875, 0.045166015625, 0.04052734375, 0.01385498046875, 0.05126953125, -0.05224609375, -0.003875732421875, 0.000701904296875, -0.06787109375, -0.051513671875, 0.0033111572265625, -0.038818359375, 0.01141357421875, 0.023193359375, 0.0123291015625, -0.0098876953125, -0.05517578125, -0.040283203125, -0.09326171875, 0.0233154296875, 0.0106201171875, 0.006378173828125, -0.04541015625, -0.01068115234375, -0.06640625, 0.039794921875, 0.0888671875, -0.0869140625, -0.0198974609375, -0.0220947265625, -0.050048828125, -0.1025390625, 0.00860595703125, 0.005950927734375, 0.0208740234375, -0.059814453125, -0.031982421875, 0.0218505859375, -0.0263671875, 0.056640625, 0.058837890625, 0.0281982421875, -0.0196533203125, -0.06494140625, -0.056640625, 0.007415771484375, -0.04150390625, 0.02197265625, -0.006988525390625, -0.001495361328125, 0.007232666015625, 0.037109375, -0.01312255859375, -0.087890625, 0.05517578125, -0.00028228759765625, -0.0164794921875, -0.009521484375, -0.039794921875, -0.0218505859375, 0.10107421875, -0.02587890625, -0.05126953125, -0.027099609375, 0.0517578125, 0.0174560546875, 0.046630859375, 0.023193359375, 0.01104736328125, -0.0267333984375, -0.044921875, 0.035400390625, 0.07763671875, 0.01300048828125, 0.021728515625, -0.00946044921875, 0.0673828125, 0.0262451171875, -0.01177978515625, -0.055419921875, 0.034912109375]
    # print("feas_map_list = ", feas_map_list)
    name = face_recognition(fea1, feas_map_list)
    print("name = ", name)
    