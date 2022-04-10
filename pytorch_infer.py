# -*- coding:utf-8 -*-
import base64
import os
import io
import chardet
from flask import Flask, request

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time

import argparse
import numpy as np
from PIL import Image
import sys
import torch


def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick]


def decode_bbox(anchors, raw_outputs, variances=[0.1, 0.1, 0.2, 0.2]):
    '''
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]
    :param anchors: numpy array with shape [batch, num_anchors, 4]
    :param raw_outputs: numpy array with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
    '''
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox


def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    '''
    generate anchors.
    :param feature_map_sizes: list of list, for example: [[40,40], [20,20]]
    :param anchor_sizes: list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
    :param anchor_ratios: list of list, for example: [[1, 0.5], [1, 0.5]]
    :param offset: default to 0.5
    :return:
    '''
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0]  # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0]  # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes


def load_pytorch_model(model_path):
    model = torch.load(model_path)
    return model


def pytorch_inference(model, img_arr):
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    device = torch.device(dev)
    model.to(device)
    input_tensor = torch.tensor(img_arr).float().to(device)
    y_bboxes, y_scores, = model.forward(input_tensor)
    return y_bboxes.detach().cpu().numpy(), y_scores.detach().cpu().numpy()


# model = load_pytorch_model('models/face_mask_detection.pth');
# model = load_pytorch_model('/Users/qi/pythonProject/JMask_Python_Server/model360.pth')
# print(sys.path[0])
os.chdir(sys.path[0])
model = load_pytorch_model('model360.pth')
# model = load_pytorch_model('FaceMaskDetection-master/models/model360.pth')
# print("model:")
# print(model)
# print("##########")
# anchor configuration
# feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    image_transposed = image_exp.transpose((0, 3, 1, 2))

    y_bboxes_output, y_cls_output = pytorch_inference(model, image_transposed)
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin - 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info


def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    idx = 0
    while status:
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(360, 360),
                      draw_result=True,
                      show_result=False)
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            # writer.write(img_raw)
            write_frame_stamp = time.time()
            idx += 1
            print("%d of %d" % (idx, total_frames))
            print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                   inference_stamp - read_frame_stamp,
                                                                   write_frame_stamp - inference_stamp))
    # writer.release()


app = Flask(__name__)


@app.route('/Mask', methods=['GET', 'POST'])
def deal():
    imgData = request.form['imgData']
    # print(imgData)
    # imgData = base64.b64decode(imgData).decode('utf-8')
    imgData = base64.b64decode(imgData)
    # print(imgData)
    # print(" 结束\n\n\n")

    # base64编码
    # img_data = base64.b64decode(imgData)

    # 转换为np数组
    img_array = np.fromstring(imgData, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img = cv2.imread("/Users/qi/Downloads/test02.jpeg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inference(img, show_result=False, target_shape=(360, 360))
    img = Image.fromarray(img)
    output = io.BytesIO()
    # img.show()
    img.save(output, format='JPEG', quality=100)
    hex_data = output.getvalue()
    # with open('/Users/qi/IdeaProjects/JMask_Server/.AppData/Web/test.jpg', mode='wb') as file:
    #     file.write(output.getvalue())
    hex_data = base64.b64encode(hex_data).decode("utf-8")
    output.close()
    # print(hex_data)
    # print(" 结束")
    return hex_data
    # return ""
# mode = request.args.get('mode')
# imgdir = request.args.get('imgdir')
# imgdir = base64.b64decode(imgdir).decode('utf-8')
# imgData = request.args.get('imgData')
# imgData = base64.b64decode(imgData).decode('utf-8')
# # print(imgdir)
# if mode == 'img':
#     img = cv2.imread(imgdir)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     inference(img, show_result=False, target_shape=(360, 360))
#     img = Image.fromarray(img)
#     output = io.BytesIO()
#     # img.show()
#     img.save(output, format='JPEG', quality=100)
#     hex_data = output.getvalue()
#     # with open('/Users/qi/IdeaProjects/JMask_Server/.AppData/Web/test.jpg', mode='wb') as file:
#     #     file.write(output.getvalue())
#     hex_data = base64.b64encode(hex_data).decode("utf-8")
#     output.close()
#     return hex_data
#     # print(hex_data)
#     # print(type(hex_data))
#     # img_data = open('/Users/qi/Downloads/images-2.jpeg', 'rb').read()
#     # response = make_response(img_data)
#     # response.headers['Content-Type'] = 'image/jpeg'
# else:
#     # base64编码
#     img_data = base64.b64decode(imgData)
#     # 转换为np数组
#     img_array = np.fromstring(img_data, np.uint8)
#     # 转换成opencv可用格式
#     img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
#     inference(img, show_result=False, target_shape=(360, 360))
#     img = Image.fromarray(img)
#     output = io.BytesIO()
#     # img.show()
#     img.save(output, format='JPEG', quality=100)
#     hex_data = output.getvalue()
#     # with open('/Users/qi/IdeaProjects/JMask_Server/.AppData/Web/test.jpg', mode='wb') as file:
#     #     file.write(output.getvalue())
#     hex_data = base64.b64encode(hex_data).decode("utf-8")
#     output.close()
#     return hex_data
#
#
#     # if imgdir == '0':
#     #     imgdir = 0
#     # run_on_video(imgdir, '', conf_thresh=0.5)
# return "Error"


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0")
    # parser = argparse.ArgumentParser(description="Face Mask Detection")
    # parser.add_argument('--img-mode', type=int, default=1, help='set 1 to run on image, 0 to run on video.')
    # parser.add_argument('--img-path', type=str, default='img/demo2.jpg', help='path to your image.')
    # parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    # args = parser.parse_args()
    # if args.img_mode:
    #     imgPath = args.img_path
    #     img = cv2.imread(imgPath)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     inference(img, show_result=True, target_shape=(360, 360))
    # else:
    #     video_path = args.video_path
    #     if args.video_path == '0':
    #         video_path = 0
    #     run_on_video(video_path, '', conf_thresh=0.5)
