from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from yolo_v8.pred import V8
from predict import Segmentation
from demo import Disparity


left_video_path='../results/apples_left.mp4'
right_video_path='../results/apples_right.mp4'

#Declare detector
detector= V8()

#Declare segmentation object
segmentation= Segmentation()

#Declare disparity object
disparity=Disparity()

tracker=YOLOv8_SORT_3D(detector=detector, segment=segmentation,disparity=disparity)

tracker.track_video(left_video_path,right_video_path, output="./IO_data/output/street_conf_0.3.mp4", show_live =False, \
                    skip_frames = 0, count_objects = True, verbose=1,frame_save_dir_path='/home/pahuja/Projects/Apple tracking/deep_sort_3D/Tests/Mahalanobis')

