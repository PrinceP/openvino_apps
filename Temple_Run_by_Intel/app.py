# Flask application to capture live webcam stream and send the frames
# to the web page that is displaying the game

# Import necessary packages
from flask import Flask, render_template, Response
import time
import cv2
import numpy as np

import logging as log
import sys

import imutils
from imutils.video import VideoStream
import pyautogui


from model_api.models import ImageModel, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage


from time import perf_counter

app = Flask(__name__)


log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

ARCHITECTURES = {
    'ae': 'HPE-assosiative-embedding',
    'higherhrnet': 'HPE-assosiative-embedding',
    'openpose': 'openpose'
}

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

coco_part_labels = [
    'nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
    'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
    'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r'
]

def draw_poses(img, poses, point_score_threshold, output_transform, skeleton=default_skeleton, draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def print_raw_results(poses, scores, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    for pose, pose_score in zip(poses, scores):
        pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
        log.debug('{} | {:.2f}'.format(pose_str, pose_score))


# generate frames and yield to Response
def gen_frames():

	next_frame_id = 1
	next_frame_id_to_show = 0

	metrics = PerformanceMetrics()
	render_metrics = PerformanceMetrics()
	plugin_config = get_user_config('CPU', '', '')
	model_adapter = OpenvinoAdapter(create_core(), "./human-pose-estimation-0007/FP16/human-pose-estimation-0007.xml", device="CPU", plugin_config=plugin_config,
								max_num_requests=0, model_parameters = {'input_layouts': None})

	start_time = perf_counter()

	# Start the video stream through the webcam
	vs = VideoStream(src=0).start()
	frame = vs.read()
	frame = cv2.flip(frame, 1)

	config = {
		'target_size': None,
		'aspect_ratio': frame.shape[1] / frame.shape[0],
		'confidence_threshold': 0.1,
		'padding_mode': 'center', 
		'delta': 0.5,
	}
	model = ImageModel.create_model(ARCHITECTURES['higherhrnet'], model_adapter, config)
	model.log_layers_info()

	hpe_pipeline = AsyncPipeline(model)
	hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})

	output_transform = OutputTransform(frame.shape[:2], None)
	
	output_resolution = (frame.shape[1], frame.shape[0])
	presenter = monitors.Presenter('', 55,
									(round(output_resolution[0] / 4), round(output_resolution[1] / 8)))

	H = frame.shape[0]
	W = frame.shape[1]
	
	# Define the boundaries
	up   =  200     
	down =  500  
	left =  400  
	right = 800 

	# By default each key press is followed by a 0.1 second pause
	pyautogui.PAUSE = 0.0

	# wait sometime until next movement is registered
	wait_time = 0.01
	start = end = 0

	# total number of frames processed thus far and skip frames
	totalFrames = 0

	# loop indefinitely
	while True:
		# grab the video frame, laterally flip it and resize it
		frame = vs.read()
		# frame = cv2.flip(frame, 1)
		frame = imutils.resize(frame, width=W)

		# initialize the action
		action = None
		
		if hpe_pipeline.callback_exceptions:
			raise hpe_pipeline.callback_exceptions[0]
		# Process all completed requests
		results = hpe_pipeline.get_result(next_frame_id_to_show)
		if results:
			(poses, scores), frame_meta = results
			frame = frame_meta['frame']
			start_time = frame_meta['start_time']

			if len(poses):
				# print_raw_results(poses, scores, next_frame_id_to_show)
				
				if poses[0][0][2] > 0.1:

					# calculate the center of the face
					centerX = int(poses[0][0][0])
					centerY = int(poses[0][0][1])

					# draw a bounding box and the center
					cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), -1)

					# determine the action
					if centerY < up:
						action = "up"
					elif centerY > down:
						action = "down"
					elif centerX < left:
						action = "left"
					elif centerX > right:
						action = "right"

			presenter.drawGraphs(frame)
			rendering_start_time = perf_counter()
			frame = draw_poses(frame, poses, 0.1, output_transform)
			render_metrics.update(rendering_start_time)
			metrics.update(start_time, frame)
			next_frame_id_to_show += 1
			end = time.time()
			# press the key
			if action is not None and end - start > wait_time:
				# print(action)
				pyautogui.press(action)
				start = time.time()

			# draw the lines
			cv2.line(frame, (0, up), (frame.shape[1], up), (255, 255, 255), 2) #UP
			cv2.line(frame, (0, down), (frame.shape[1], down), (255, 255, 255), 2) #DOWN
			cv2.line(frame, (left, up), (left, down), (255, 255, 255), 2) #LEFT
			cv2.line(frame, (right, up), (right, down), (255, 255, 255), 2) #RIGHT

			# increment the totalFrames and draw the action on the frame
			totalFrames += 1
			text = "{}: {}".format("Action", action)
			cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

			# Generate a stream of frame bytes
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		if hpe_pipeline.is_ready():
			# Get new image/frame
			start_time = perf_counter()
			frame = vs.read()
			if frame is None:
				break

			# Submit for inference
			hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
			next_frame_id += 1

		else:
			# Wait for empty request
			hpe_pipeline.await_any()

	hpe_pipeline.await_all()
	if hpe_pipeline.callback_exceptions:
		raise hpe_pipeline.callback_exceptions[0]
	# Process completed requests
	for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
		results = hpe_pipeline.get_result(next_frame_id_to_show)
		(poses, scores), frame_meta = results
		frame = frame_meta['frame']
		start_time = frame_meta['start_time']

		if len(poses):
			# print_raw_results(poses, scores, next_frame_id_to_show)
			if poses[0][0][2] > 0.1:

					# calculate the center of the face
					centerX = int(poses[0][0][0])
					centerY = int(poses[0][0][1])

					# draw a bounding box and the center
					cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), -1)

					# determine the action
					if centerY < up:
						action = "up"
					elif centerY > down:
						action = "down"
					elif centerX < left:
						action = "left"
					elif centerX > right:
						action = "right"
			end = time.time()
			# press the key
			if action is not None and end - start > wait_time:
				# print(action)
				pyautogui.press(action)
				start = time.time()

			# draw the lines
			cv2.line(frame, (0, up), (frame.shape[1], up), (255, 255, 255), 2) #UP
			cv2.line(frame, (0, down), (frame.shape[1], down), (255, 255, 255), 2) #DOWN
			cv2.line(frame, (left, up), (left, down), (255, 255, 255), 2) #LEFT
			cv2.line(frame, (right, up), (right, down), (255, 255, 255), 2) #RIGHT

			# increment the totalFrames and draw the action on the frame
			totalFrames += 1
			text = "{}: {}".format("Action", action)
			cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

			# Generate a stream of frame bytes
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		presenter.drawGraphs(frame)
		rendering_start_time = perf_counter()
		frame = draw_poses(frame, poses, 0.1, output_transform)
		render_metrics.update(rendering_start_time)
		metrics.update(start_time, frame)
		

	metrics.log_total()




# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Video streaming route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)