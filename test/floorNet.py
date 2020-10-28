import time
import cv2 as cv
import numpy as np

from utils import ros, image, odometry
from src.floorNet import FloorNet


def ml():
    # Init modules
    floorNet = FloorNet()
    camera = cv.VideoCapture(0)
    angles = odometry.Angle()

    # Prediction
    while True:
        start = time.time()
        print("======================================")
        # Get images
        _, frame = camera.read()
        print('*** Debug camera shape:', frame.shape)

        # Infer
        img, mask = floorNet.predict(frame)
        img = (img*127.5+127.5)/255

        # Visualization
        mask, clusters = image.kmeans(mask)
        [y, x] = clusters[0] - clusters[1]
        angles.push(np.arctan(y/x)*180/np.pi)
        diff_angle = angles.mean()
        print(diff_angle)
        img = cv.addWeighted(mask, 0.5, img, 0.5, 0)
        img = cv.resize(img, (512, 512))
        cv.imshow('Video', img)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        # Calculate frames per second (FPS)
        end = time.time()
        fps = 1/(end-start)
        print('Total estimated time: {:.4f}'.format(end-start))
        print("FPS: {:.1f}".format(fps))


def infer(botshell, debug=False):
    # Init modules
    floorNet = FloorNet()
    angles = odometry.Angle()
    odo = odometry.Odometry(botshell, floorNet.image_shape)
    if debug:
        rosimg = ros.ROSImage()
        talker = rosimg.gen_talker('/ocp/draw_image/compressed')
    camera = cv.VideoCapture(1)

    # Prediction
    while True:
        start = time.time()
        print("======================================")
        # Get images
        _, frame = camera.read()
        print('*** Debug camera shape:', frame.shape)

        # Infer
        img, mask = floorNet.predict(frame)
        img = (img*127.5+127.5)/255
        mask, clusters = image.kmeans(mask)
        [y, x] = clusters[0] - clusters[1]
        angles.push(np.arctan(y/x)*180/np.pi)
        diff_angle = angles.mean()

        print('*** Debug differential angle:', diff_angle)
        if diff_angle < -5:
            odo.move((0, 100))
        elif diff_angle > 5:
            odo.move((100, 0))
        else:
            odo.move((100, 100))

        # Visualize
        if debug:
            img = cv.addWeighted(mask, 0.5, img, 0.5, 0)
            img = img * 255
            talker.push(img)

        # Calculate frames per second (FPS)
        end = time.time()
        fps = 1/(end-start)
        print('Total estimated time: {:.4f}'.format(end-start))
        print("FPS: {:.1f}".format(fps))

    talker.stop()
    rosimg.stop()
