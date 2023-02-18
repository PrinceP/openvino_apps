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

#OpenVINO
import base64
import logging as log
import sys
from time import perf_counter
from argparse import ArgumentParser
from pathlib import Path
import cv2

from model_api.models import MaskRCNNModel, YolactModel, OutputTransform
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.performance_metrics import PerformanceMetrics


import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import InstanceSegmentationVisualizer

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

def get_model(model_adapter, configuration):
    inputs = model_adapter.get_input_layers()
    outputs = model_adapter.get_output_layers()
    if len(inputs) == 1 and len(outputs) == 4 and 'proto' in outputs.keys():
        return YolactModel(model_adapter, configuration)
    return MaskRCNNModel(model_adapter, configuration)


def print_raw_results(boxes, classes, scores, frame_id):
    log.info('  -------------------------- Frame # {} --------------------------  '.format(frame_id))
    log.info('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
    for box, cls, score in zip(boxes, classes, scores):
        log.info('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

def load_model():
    plugin_config = get_user_config('CPU', '', None)
    model_adapter = OpenvinoAdapter(create_core(), './instance-segmentation-person-0007/FP16/instance-segmentation-person-0007.xml', device='CPU', plugin_config=plugin_config,
                                    max_num_requests=0,
                                    model_parameters={'input_layouts': None})
    configuration = {
        'confidence_threshold': 0.5,
        'path_to_labels': './instance-segmentation-person-0007/labels.txt',
    }
    model = get_model(model_adapter, configuration)
    model.log_layers_info()
    pipeline = AsyncPipeline(model)
    visualizer = InstanceSegmentationVisualizer('./instance-segmentation-person-0007/labels.txt')
    if pipeline.is_ready():
        return pipeline, visualizer

def image_to_crop(frame, pipeline, visualizer):
    start_time = perf_counter()
    
    pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    
    
    # if pipeline.callback_exceptions:
    #     raise pipeline.callback_exceptions[0]
    # Process all completed requests
    results = pipeline.get_result(0)
    if results:
        (scores, classes, boxes, masks), frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']
        print_raw_results(boxes, classes, scores, 0)
        frame = visualizer(frame, boxes, classes, scores, masks, None)
    else:
        results = pipeline.get_result(0)
        if results:
            (scores, classes, boxes, masks), frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            print_raw_results(boxes, classes, scores, 0)
            frame = visualizer(frame, boxes, classes, scores, masks, None)
        
    return frame

def main(page: Page):
    page.title = "Intel Lens"
    page.theme_mode = "light"
    page.padding = 50
    page.auto_scroll = True

    page.window_height = 1200
    page.window_width = 1000
    
    page.update()
    
    pipeline, visualizer  = load_model()

    # Pick files dialog
    def pick_files_result(e: FilePickerResultEvent):
        selected_files.src = (
            "".join(map(lambda f: f.path, e.files)) if e.files else "Cancelled!"
        )
        log.debug(selected_files.src)
        frame = cv2.imread(selected_files.src)
        processed_frame = image_to_crop(frame, pipeline, visualizer)
        filler_text.value = "PERSON"

        processed_file.src_base64 = base64.b64encode(cv2.imencode('.jpg', processed_frame)[1]).decode()
        selected_files.src_base64 = base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode()
        processed_file.update()
        selected_files.update()
        filler_text.update()

    pick_files_dialog = FilePicker(on_result=pick_files_result)
    selected_files = Image(src_base64="1234", fit=ImageFit.CONTAIN)
    processed_file = Image(src_base64="1234", fit=ImageFit.CONTAIN)
    page.overlay.append(pick_files_dialog)

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