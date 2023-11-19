#! /usr/bin/env python3

from  bridge_wrapper import *
from yolo_v8.pred import V8
from predict import Segmentation
from demo import Disparity
import rospy
from sensor_msgs.msg import Image
import message_filters

params = {
    "skip_frames": 0,
    "show_live": False,
    "count_objects": False,
    "verbose": 0,
    "frame_save_dir_path": './Frames',
    "output": False,
    "debug": False,
}

tracker = None
frame_num = 0
result_pub = None

def read_cameras():
    imageL = message_filters.Subscriber("/theia/left/image_raw", Image)
    imageR = message_filters.Subscriber("/theia/right/image_raw", Image)

    # Synchronize images
    ts = message_filters.ApproximateTimeSynchronizer([imageL, imageR], queue_size=10, slop=0.5)
    ts.registerCallback(image_callback)
    rospy.spin()

def image_callback(imageL, imageR):
    # br = CvBridge()
    rospy.loginfo("receiving frame")
    frame_left = np.frombuffer(imageL.data, dtype=np.uint8).reshape(imageL.height, imageL.width, -1)
    frame_right = np.frombuffer(imageR.data, dtype=np.uint8).reshape(imageR.height, imageR.width, -1)

    global frame_num, params, tracker
    frame_num +=1

    #Downsample image
    # frame_left=cv2.resize(frame_left,dsize=(800,640),interpolation=cv2.INTER_LANCZOS4)
    # frame_right=cv2.resize(frame_right,dsize=(800,640),interpolation=cv2.INTER_LANCZOS4)

    if params["skip_frames"] and not frame_num % params["skip_frames"]: return # skip every nth frame. When every frame is not important, you can use this to fasten the process
    if params["verbose"] >= 1:start_time = time.time()

    ############################  Detection Model #############################################
    yolo_dets,scores,_=tracker.detector.pred(frame_left.copy(),debug=False)      
                
    ############################ Find 3D information ###########################################
    disparity=tracker.disparity.find_disparity(frame_left, frame_right)           
    seg_mask=tracker.segment.predict_img(cv2.cvtColor(frame_left.copy(),cv2.COLOR_BGR2RGB))

    image, point_mask,yolo_dets,points_2D=ut.find_center(yolo_dets, frame_left.copy(), seg_mask, debug=params["debug"])
    
    '''
    Image: Left frame showing fruit center and bounding boxes if debug"=True        
    '''
    if params["debug"]:
        cv2.namedWindow("Fruit center & Bounding Box",cv2.WINDOW_NORMAL)
        cv2.namedWindow("Only Fruit",cv2.WINDOW_NORMAL)

        print(frame_left.shape,seg_mask.shape)
        image_added=cv2.addWeighted(frame_left,1.0,cv2.cvtColor(seg_mask,cv2.COLOR_BGR2RGB),0.5,0.0)

        cv2.imshow("Only Fruit",image_added)
        cv2.imshow("Fruit center & Bounding Box",image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    disparity=cv2.resize(disparity,dsize=(point_mask.shape[1],point_mask.shape[0]),interpolation=cv2.INTER_LANCZOS4)

    points_3d=ut.obtain_3d_volume(disparity,frame_left.copy(),point_mask=point_mask,fruit_mask=seg_mask,points_2D=points_2D,\
                                    save_file=True, frame_num=frame_num)

    if frame_num!=1: #Update rover coordinates only from 2nd frame
        points_3d=tracker.rover_detec(tracker.base_cord)+points_3d #Shifting 3D cordinates from rover to world origin

    else: 
        points_3d=tracker.base_cord+points_3d

    names=np.array(["Apple"]*len(yolo_dets))

    # import ipdb; ipdb.set_trace()
    if len(points_3d)!=len(yolo_dets): #Condition to check if all the bounding boxes have a fruit
        print("*"*20,"STOPPING CODE","*"*20)
        print("All boxes do not have a center point")
        return
            
    if params["count_objects"]:
        cv2.putText(frame_left, "Objects being tracked: {}".format(len(names)), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255), 2)

    ######################################### SORT_3D ##############################################

    position_3D=np.vstack((tracker.base_cord,points_3d))  #Saving measured objects: rover and apples in one data structure
    yolo_dets=np.vstack((np.array([0,0,0,0]),yolo_dets)) # Making 2d location of the apples of the similar structure, i.e. apple location start from pos=1
    scores=np.hstack((np.array([0]),scores))            #Detection scores; Concatenting 0 at pos=0 so have similar structure
    detections = Detection(yolo_dets, scores, names,  position_3D) # detection object for rover and all the apples

    if frame_num!=1:
        tracker.tracker.predict()  # Call the tracker except for frame 1 since tracker object of detections is made at the end of frame 1

    matches=tracker.tracker.update(detections) #  update using Kalman Gain

    # Visualizing           
    cmap = plt.get_cmap('tab20b') #initialize color map
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]   
    
    if len(matches)!=0:
        matches=np.vstack((matches)) #Add matched tracks to the datastructure

    for match in matches:  # update new findings AKA tracks                
        if tracker.tracker.tracks.is_confirmed([match[0]])==False:
            # import ipdb; ipdb.set_trace()
            continue 
        bbox = detections.points_2D[match[1]]
        class_name = 'Apple'

        color = colors[int(tracker.tracker.tracks.track_id[match[0]]) % len(colors)]  # draw bbox on screen
        color = [i * 255 for i in color]
        text_color=(255,255,255)

        cv2.rectangle(frame_left, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame_left, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(tracker.tracker.tracks.track_id[match[0]])))*10, int(bbox[1])), color, -1) #To make a solid rectangle box to write text on
        cv2.putText(frame_left, class_name + ":" + str(tracker.tracker.tracks.track_id[match[0]]),(int(bbox[0]), int(bbox[1]-11)),0, 0.5, (text_color),2, lineType=cv2.LINE_AA)
        # cv2.putText(frame_left, class_name + " " + str(track.track_id)+':'+str(round(ut.occlusion_score(bbox,mask),3)),(int(bbox[0]), int(bbox[1]-11)),0, 0.8, (text_color),2, lineType=cv2.LINE_AA)  
        cv2.putText(frame_left, "Frame_num:"+str(frame_num),(len(frame_left[0])-100,len(frame_left)-50),0, 0.5, (255,255,255),2, lineType=cv2.LINE_AA)
    
        if params["verbose"] == 2:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(tracker.tracker.tracks.track_id[match[0]]), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            
    # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------

    if params["verbose"] >= 1:
        fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
        if not params["count_objects"]: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
        else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {len(tracker.tracker.tracks.mean_2D)}")
    
    result = np.asarray(frame_left)
    result = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)
    
    if params["output"]: out.write(result) # save output video

    if params["show_live"]:
        cv2.namedWindow("Output Video",cv2.WINDOW_NORMAL)
        cv2.imshow("Output Video", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): return
    
    msg = Image()
    msg.height = result.shape[0]
    msg.width = result.shape[1]
    msg.encoding = "bgr8"
    msg.data = result.tobytes()
    msg.step = len(msg.data) // msg.height
    msg.header.stamp = rospy.Time.now()
    result_pub.publish(msg)
    # frame_left = np.frombuffer(imageL.data, dtype=np.uint8).reshape(imageL.height, imageL.width, -1)


if __name__ == '__main__':
    rospy.init_node('tracker_node')

    result_pub = rospy.Publisher('/tracker_node/result', Image, queue_size=10)

    #Declare detector
    detector= V8(conf=0.1, iou=0.15) #change values of confidence and iou to tune the detector

    #Declare segmentation object
    segmentation= Segmentation()

    #Declare disparity object
    disparity=Disparity()

    tracker=YOLOv8_SORT_3D(detector=detector, rover_coor_path='../results/rtk_fix.csv',segment=segmentation,disparity=disparity)

    try:
        read_cameras()
    except rospy.ROSInterruptException:
        pass

# tracker.track_video(left_video_path,right_video_path,output="./IO_data/output/street_conf_0.3.mp4", params["show_live"] =True, \
#                     params["skip_frames"] = 0, params["count_objects"] = True, params["verbose"]=1,params["frame_save_dir_path"]='./Frames')

