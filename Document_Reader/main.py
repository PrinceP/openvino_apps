
import logging as log
import sys
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

import base64
import cv2
import numpy as np
from scipy.special import softmax
from openvino.runtime import Core, get_version

from visualizers import InstanceSegmentationVisualizer

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28




from typing import Dict
import flet
from flet import (
    Column,
    FilePicker,
    FilePickerResultEvent,
    Page,
    Row,
    Text,
    icons,
    Image,
    Container,
    alignment,
    colors,
    IconButton,
    CircleAvatar,
    TextThemeStyle,
    ScrollMode,
    ImageFit,
)

def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp


def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


def load_model():
    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    # Read IR
    log.info('Reading Mask-RCNN model {}'.format("./text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml"))
    mask_rcnn_model = core.read_model("./text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml")

    input_tensor_name = 'image'
    try:
        n, c, h, w = mask_rcnn_model.input(input_tensor_name).shape
        if n != 1:
            raise RuntimeError('Only batch 1 is supported by the demo application')
    except RuntimeError:
        raise RuntimeError('Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))
    
    required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
    for output_tensor_name in required_output_names:
        try:
            mask_rcnn_model.output(output_tensor_name)
        except RuntimeError:
            raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
                ', '.join(required_output_names)))

    log.info('Reading Text Recognition Encoder model {}'.format("./text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml"))
    text_enc_model = core.read_model("./text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml")

    log.info('Reading Text Recognition Decoder model {}'.format("./text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-decoder.xml"))
    text_dec_model = core.read_model("./text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml")

    mask_rcnn_compiled_model = core.compile_model(mask_rcnn_model, device_name="CPU")
    mask_rcnn_infer_request = mask_rcnn_compiled_model.create_infer_request()

    text_enc_compiled_model = core.compile_model(text_enc_model, "CPU")
    text_enc_output_tensor = text_enc_compiled_model.outputs[0]
    text_enc_infer_request = text_enc_compiled_model.create_infer_request()
    
    text_dec_compiled_model = core.compile_model(text_dec_model, "CPU")
    text_dec_infer_request = text_dec_compiled_model.create_infer_request()

    hidden_shape = text_dec_model.input("prev_hidden").shape
    text_dec_output_names = {"output", "hidden"}

    visualizer = InstanceSegmentationVisualizer(show_boxes="store_true", show_scores=None)

    return input_tensor_name, required_output_names, h, w, c, text_enc_output_tensor, mask_rcnn_infer_request, text_enc_infer_request, text_dec_infer_request, hidden_shape, text_dec_output_names, visualizer    

def image_to_text(frame, input_tensor_name, required_output_names, h, w, c, text_enc_output_tensor, mask_rcnn_infer_request, text_enc_infer_request, text_dec_infer_request, hidden_shape, text_dec_output_names, visualizer):
    # log.info('OpenVINO Runtime')
    # log.info('\tbuild: {}'.format(get_version()))
    # core = Core()

    # # Read IR
    # log.info('Reading Mask-RCNN model {}'.format("./text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml"))
    # mask_rcnn_model = core.read_model("./text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml")

    # input_tensor_name = 'image'
    # try:
    #     n, c, h, w = mask_rcnn_model.input(input_tensor_name).shape
    #     if n != 1:
    #         raise RuntimeError('Only batch 1 is supported by the demo application')
    # except RuntimeError:
    #     raise RuntimeError('Demo supports only topologies with the following input tensor name: {}'.format(input_tensor_name))
    
    # required_output_names = {'boxes', 'labels', 'masks', 'text_features'}
    # for output_tensor_name in required_output_names:
    #     try:
    #         mask_rcnn_model.output(output_tensor_name)
    #     except RuntimeError:
    #         raise RuntimeError('Demo supports only topologies with the following output tensor names: {}'.format(
    #             ', '.join(required_output_names)))

    # log.info('Reading Text Recognition Encoder model {}'.format("./text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml"))
    # text_enc_model = core.read_model("./text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml")

    # log.info('Reading Text Recognition Decoder model {}'.format("./text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-decoder.xml"))
    # text_dec_model = core.read_model("./text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml")

    # mask_rcnn_compiled_model = core.compile_model(mask_rcnn_model, device_name="CPU")
    # mask_rcnn_infer_request = mask_rcnn_compiled_model.create_infer_request()

    # text_enc_compiled_model = core.compile_model(text_enc_model, "CPU")
    # text_enc_output_tensor = text_enc_compiled_model.outputs[0]
    # text_enc_infer_request = text_enc_compiled_model.create_infer_request()
    
    # text_dec_compiled_model = core.compile_model(text_dec_model, "CPU")
    # text_dec_infer_request = text_dec_compiled_model.create_infer_request()

    # hidden_shape = text_dec_model.input("prev_hidden").shape
    # text_dec_output_names = {"output", "hidden"}

    # visualizer = InstanceSegmentationVisualizer(show_boxes="store_true", show_scores=None)

    if frame is not None:
        if not "store_true":
            # Resize the image to a target size.
            scale_x = w / frame.shape[1]
            scale_y = h / frame.shape[0]
            input_image = cv2.resize(frame, (w, h))
        else:
            # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
            scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
            input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                           (0, w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((1, c, h, w)).astype(np.float32)

        # Run the MaskRCNN model.
        mask_rcnn_infer_request.infer({input_tensor_name: input_image})
        outputs = {name: mask_rcnn_infer_request.get_tensor(name).data[:] for name in required_output_names}

        # Parse detection results of the current request
        boxes = outputs['boxes'][:, :4]
        scores = outputs['boxes'][:, 4]
        classes = outputs['labels'].astype(np.uint32)
        raw_masks = outputs['masks']
        text_features = outputs['text_features']

        # Filter out detections with low confidence.
        detections_filter = scores > 0.5
        scores = scores[detections_filter]
        classes = classes[detections_filter]
        boxes = boxes[detections_filter]
        raw_masks = raw_masks[detections_filter]
        text_features = text_features[detections_filter]

        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        masks = []
        for box, cls, raw_mask in zip(boxes, classes, raw_masks):
            mask = segm_postprocess(box, raw_mask, frame.shape[0], frame.shape[1])
            masks.append(mask)

        texts = []
        for feature in text_features:
            input_data = {'input': np.expand_dims(feature, axis=0)}
            feature = text_enc_infer_request.infer(input_data)[text_enc_output_tensor]
            feature = np.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            text_confidence = 1.0
            for i in range(MAX_SEQ_LEN):
                text_dec_infer_request.infer({
                    "prev_symbol": np.reshape(prev_symbol_index, (1,)),
                    "prev_hidden": hidden,
                    "encoder_outputs": feature})
                decoder_output = {name: text_dec_infer_request.get_tensor(name).data[:] for name in text_dec_output_names}
                symbols_distr = decoder_output["output"]
                symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                text_confidence *= symbols_distr_softmaxed[prev_symbol_index]
                if prev_symbol_index == EOS_INDEX:
                    break
                text += '  abcdefghijklmnopqrstuvwxyz0123456789'[prev_symbol_index]
                hidden = decoder_output["hidden"]

            texts.append(text if text_confidence >= 0.5 else '')

        if len(boxes):
            log.debug('  -------------------------- Frame # {} --------------------------  '.format(1))
            log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
            for box, cls, score, mask in zip(boxes, classes, scores, masks):
                log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

        frame = visualizer(frame, boxes, classes, scores, masks, None, texts)

        return texts, frame

def main(page: Page):
    page.title = "Intel Lens"
    page.theme_mode = "light"
    page.padding = 50
    page.auto_scroll = True

    page.window_height = 1500
    page.window_width = 1800
    
    page.update()
    input_tensor_name, required_output_names, h, w, c, text_enc_output_tensor, mask_rcnn_infer_request, text_enc_infer_request, text_dec_infer_request, hidden_shape, text_dec_output_names, visualizer  = load_model()

    # Pick files dialog
    def pick_files_result(e: FilePickerResultEvent):
        selected_files.src = (
            "".join(map(lambda f: f.path, e.files)) if e.files else "Cancelled!"
        )
        log.debug(selected_files.src)
        frame = cv2.imread(selected_files.src)
        texts, processed_frame = image_to_text(frame, input_tensor_name, required_output_names, h, w, c, text_enc_output_tensor, mask_rcnn_infer_request, text_enc_infer_request, text_dec_infer_request, hidden_shape, text_dec_output_names, visualizer)
        text_inside_image.value = " ".join(t.capitalize() for t in texts)
        filler_text.value = "Copy the following text.."

        processed_file.src_base64 = base64.b64encode(cv2.imencode('.jpg', processed_frame)[1]).decode()
        selected_files.src_base64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
        processed_file.update()
        selected_files.update()
        text_inside_image.update()
        filler_text.update()

    pick_files_dialog = FilePicker(on_result=pick_files_result)
    selected_files = Image(src_base64="1234", fit=ImageFit.CONTAIN)
    processed_file = Image(src_base64="1234", fit=ImageFit.CONTAIN)
    page.overlay.append(pick_files_dialog)

    text_inside_image = Text()
    filler_text = Text()

    page.add(Column(
                    [   
                        Container(
                            content=Text("INTEL LENS", 
                                        style=TextThemeStyle.DISPLAY_SMALL,
                                        color=colors.WHITE),
                            margin=10,
                            padding=5,
                            alignment=alignment.center,
                            bgcolor=colors.BLUE,
                            border_radius=10,
                        ),
                        
                        Container(
                            content=IconButton(
                                icon=icons.CAMERA,
                                icon_color="blue400",
                                icon_size=50,
                                tooltip="Pick a image",
                                on_click=lambda _: pick_files_dialog.pick_files(
                                        allow_multiple=False,
                                        allowed_extensions=["jpg", "jpeg", "png"],
                                    ),
                                ),
                            margin=10,
                            padding=5,
                            alignment=alignment.center,
                            border_radius=10,
                        ),
                        
                        
                        Row([  
                                Container(
                                    content=selected_files,
                                    margin=10,
                                    padding=10,
                                    alignment=alignment.center_left,
                                    border_radius=4,
                                ),

                                Container(
                                    content=processed_file,
                                    margin=10,
                                    padding=10,
                                    alignment=alignment.center_right,
                                    border_radius=4,
                                ),
                        ]),

                        Container(
                            content=filler_text,
                            margin=10,
                            padding=5,
                            alignment=alignment.center,
                            border_radius=10,
                        ),
                        
                        Container(
                            content=text_inside_image,
                            margin=10,
                            padding=5,
                            alignment=alignment.center,
                            border_radius=10,
                        ),


                        Row([
                                Text("Made by", font_family="RobotoSlab",),
                                CircleAvatar(
                                    foreground_image_url="https://avatars.githubusercontent.com/u/10251537?s=96&v=4",
                                ),
                                Text("Using", font_family="RobotoSlab",),
                                CircleAvatar(
                                    foreground_image_url="https://avatars.githubusercontent.com/u/17888862?s=200&v=4",
                                    radius=25
                                ),
                            ]
                            ),
                    ],
                scroll=ScrollMode.AUTO,
            )
    )

    page.update()


flet.app(target=main, assets_dir="uploads")