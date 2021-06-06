
import streamlit as st
import numpy as np
import cv2

from PIL import Image
import tempfile
from pathlib import Path


st.title("object detection")
st.write("select souce of file "
         )

modelname = st.sidebar.selectbox(
    'Select model',
    ('maskrnn', 'ssd')
)

webcam=st.button("webcam")

st.write("webcam status :"+str("ON"if webcam else "OFF"))



#print(webcam,"webcam")
file=st.file_uploader("upload video or image",type=['png','jpeg','jpg','mp4'])

if modelname=='ssd':
    #f=open("MobileNetSSD_deploy.caffemodel")
    #f2=open("MobileNetSSD_deploy.prototxt")
    if Path("MobileNetSSD_deploy.caffemodel").is_file() and Path("MobileNetSSD_deploy.prototxt").is_file():
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


        netssd = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
        st.write("ssd loaded")
        conf=st.slider("confidence value",min_value=0.00,max_value=1.00,value=0.20,step=0.05)


        if webcam or (file!=None and file.type[0:5]=='video'):
            if webcam:
                vf=cv2.VideoCapture(0)
            elif file.type[0:5]=='video':
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file.read())
                vf = cv2.VideoCapture(tfile.name)
            img=np.array(np.ones((600,650,3)))
            k=st.image(img,use_column_width=True,clamp = True)

            stop=st.button("stop")
            while True:
                (f,frame)=vf.read()
                if not f:
                    break

        # resize the frame, grab the frame dimensions, and convert it to
        # a blob
                frame = cv2.resize(frame, (frame.shape[0],400))
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

                # pass the blob through the network and obtain the detections and
                # predictions
                netssd.setInput(blob)
                detections = netssd.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > conf:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                     confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        frame=cv2.resize(frame,(h,w))
                        k.image(frame)
                    if stop:
                        webcam=0
        elif not webcam and file!=None:


            #print(file,"here1")
            if file.type[0:5] =="image":
                file_details={"Filename":file.name,"file type:":file.type}
                st.write(file_details)
                img=Image.open(file)
                img1=np.array(np.ones((600,650,3)))
                k=st.image(img1,use_column_width=True,clamp = True)
                st.write(file.name)
                frame=np.array(img)
                frame = cv2.resize(frame, (frame.shape[0],400))
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

                # pass the blob through the network and obtain the detections and
                # predictions
                netssd.setInput(blob)
                detections = netssd.forward()
                if not len(np.arange(0, detections.shape[2])):
                    #print(len(np.arange(0, detections.shape[2])))
                    k.image(frame)
                    st.write("no object in threshold")
                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > conf:
                        # extract the index of the class label from the
                        # `detections`, then compute the (x, y)-coordinates of
                        # the bounding box for the object
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the prediction on the frame
                        label = "{}: {:.2f}%".format(CLASSES[idx],
                                                     confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        st.write(label)

                        k.image(frame)
                        #print("frame got")
            else :
                st.write("unsuported file format "
                         "supported file formats are 'png','jpeg','jpg' ")



    else:
        st.write("MobileNetSSD_deploy.caffemodel and or MobileNetSSD_deploy.prototxt couldn't find" )

if modelname=='maskrnn':
    #f=open("MobileNetSSD_deploy.caffemodel")
    #f2=open("MobileNetSSD_deploy.prototxt")
    labelsPath = "object_detection_classes_coco.txt"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath ="frozen_inference_graph.pb"
    configPath ="mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    if Path("frozen_inference_graph.pb").is_file() and Path("mask_rcnn_inception_v2_coco_2018_01_28.pbtxt").is_file():
        netmask=cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
        print(netmask,"network")


        st.write("maskrnn loaded")
        conf=st.slider("confidence value",min_value=0.00,max_value=1.00,value=0.20,step=0.05)
        thresh=st.slider("threshold value",min_value=0.00,max_value=1.00,value=0.20,step=0.05)

        if webcam or (file!=None and file.type[0:5]=='video'):
            if webcam:
                vf=cv2.VideoCapture(0)
            elif file.type[0:5]=='video':
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(file.read())
                vf = cv2.VideoCapture(tfile.name)

            img=np.array(np.ones((600,650,3)))
            k=st.image(img,use_column_width=True,clamp = True)

            stop=st.button("stop")
            while True:
        # read the next frame from the file
                (grabbed, frame) = vf.read()
                #print(frame)
                # if the frame was not grabbed, then we have reached the end
                # of the stream
                if not grabbed:
                    break

                # construct a blob from the input frame and then perform a
                # forward pass of the Mask R-CNN, giving us (1) the bounding box
                # coordinates of the objects in the image along with (2) the
                # pixel-wise segmentation for each specific object
                blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
                netmask.setInput(blob)
                print("blob",blob)
                (boxes, masks) = netmask.forward(["detection_out_final",
                                              "detection_masks"])

                # loop over the number of detected objects
                for i in range(0, boxes.shape[2]):
                    # extract the class ID of the detection along with the
                    # confidence (i.e., probability) associated with the
                    # prediction
                    classID = int(boxes[0, 0, i, 1])
                    confidence = boxes[0, 0, i, 2]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > conf:
                        # scale the bounding box coordinates back relative to the
                        # size of the frame and then compute the width and the
                        # height of the bounding box
                        (H, W) = frame.shape[:2]
                        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")
                        boxW = endX - startX
                        boxH = endY - startY

                        # extract the pixel-wise segmentation for the object,
                        # resize the mask such that it's the same dimensions of
                        # the bounding box, and then finally threshold to create
                        # a *binary* mask
                        mask = masks[i, classID]
                        mask = cv2.resize(mask, (boxW, boxH),
                                          interpolation=cv2.INTER_CUBIC)
                        mask = (mask > thresh)

                        # extract the ROI of the image but *only* extracted the
                        # masked region of the ROI
                        roi = frame[startY:endY, startX:endX][mask]

                        # grab the color used to visualize this particular class,
                        # then create a transparent overlay by blending the color
                        # with the ROI
                        color = COLORS[classID]
                        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

                        # store the blended ROI in the original frame
                        frame[startY:endY, startX:endX][mask] = blended

                        # draw the bounding box of the instance on the frame
                        color = [int(c) for c in color]
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      color, 2)

                        # draw the predicted label and associated probability of
                        # the instance segmentation on the frame
                        text = "{}: {:.4f}".format(LABELS[classID], confidence)
                        cv2.putText(frame, text, (startX, startY - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        k.image(frame)
                    if stop:
                        webcam=0
        elif not webcam and file!=None :


            #print(file,"here1")
            if file.type[0:5] =="image":
                file_details={"Filename":file.name,"file type:":file.type}
                st.write(file_details)
                img=Image.open(file)
                img1=np.array(np.ones((600,650,3)))
                k=st.image(img1,use_column_width=True,clamp = True)
                st.write(file.name)
                frame=np.array(img)
                blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
                netmask.setInput(blob)
                (boxes, masks) = netmask.forward(["detection_out_final",
                                                  "detection_masks"])

                # loop over the number of detected objects
                for i in range(0, boxes.shape[2]):
                    # extract the class ID of the detection along with the
                    # confidence (i.e., probability) associated with the
                    # prediction
                    classID = int(boxes[0, 0, i, 1])
                    confidence = boxes[0, 0, i, 2]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > conf:
                        # scale the bounding box coordinates back relative to the
                        # size of the frame and then compute the width and the
                        # height of the bounding box
                        (H, W) = frame.shape[:2]
                        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")
                        boxW = endX - startX
                        boxH = endY - startY

                        # extract the pixel-wise segmentation for the object,
                        # resize the mask such that it's the same dimensions of
                        # the bounding box, and then finally threshold to create
                        # a *binary* mask
                        mask = masks[i, classID]
                        mask = cv2.resize(mask, (boxW, boxH),
                                          interpolation=cv2.INTER_CUBIC)
                        mask = (mask > thresh)

                        # extract the ROI of the image but *only* extracted the
                        # masked region of the ROI
                        roi = frame[startY:endY, startX:endX][mask]

                        # grab the color used to visualize this particular class,
                        # then create a transparent overlay by blending the color
                        # with the ROI
                        color = COLORS[classID]
                        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

                        # store the blended ROI in the original frame
                        frame[startY:endY, startX:endX][mask] = blended

                        # draw the bounding box of the instance on the frame
                        color = [int(c) for c in color]
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      color, 2)

                        # draw the predicted label and associated probability of
                        # the instance segmentation on the frame
                        text = "{}: {:.4f}".format(LABELS[classID], confidence)
                        cv2.putText(frame, text, (startX, startY - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        k.image(cv2.cvtColor(frame,"blue"))
                        st.write(text)

                        #print("frame got")
            else :
                st.write("unsuported file format "
                         "supported file formats are 'png','jpeg','jpg','mp4' ")



    else:
        st.write("frozen_inference_graph.pb mask_rcnn_inception_v2_coco_2018_01_28.pbtxt couldn't find" )







    #st.write(frame)






