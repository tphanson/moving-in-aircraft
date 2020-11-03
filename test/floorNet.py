import time
import cv2 as cv
import numpy as np

from utils import ros, image, odometry
from src.floorNet import FloorNet


def segment(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x_0 = a * rho
    y_0 = b * rho
    x_1 = int(x_0 + 1000 * (-b))
    y_1 = int(y_0 + 1000 * (a))
    x_2 = int(x_0 - 1000 * (-b))
    y_2 = int(y_0 - 1000 * (a))

    return ((x_1, y_1), (x_2, y_2))


def detect_edge(_):
    camera = cv.VideoCapture(0)
    rosimg = ros.ROSImage()
    talker = rosimg.gen_talker('/ocp/draw_image/compressed')

    while True:
        start = time.time()
        print("======================================")
        # Get images
        _, frame = camera.read()
        print('*** Debug camera shape:', frame.shape)

        img = cv.resize(frame, (512, 512))
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        blur = cv.GaussianBlur(gray, (11, 11), 0)
        canny = cv.Canny(blur, 50, 150)
        hough = cv.HoughLinesP(
            canny, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=100)
        lines = np.squeeze(hough)
        for line in lines:
            print(line)
        print('=============================')

        img = cv.cvtColor(canny, cv.COLOR_GRAY2RGB)
        talker.push(img)

        # Calculate frames per second (FPS)
        end = time.time()
        fps = 1/(end-start)
        if end-start < 0.1:
            time.sleep(0.1 - (end-start))
        print('Total estimated time: {:.4f}'.format(end-start))
        print("FPS: {:.1f}".format(fps))


def cluster():
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
        diff_angle = np.arctan(y/x)*180/np.pi
        # angles.push(np.arctan(y/x)*180/np.pi)
        # diff_angle = angles.mean()

        print('*** Debug differential angle:', diff_angle)
        if diff_angle < -2:
            odo.move((-25, -25))
        elif diff_angle > 2:
            odo.move((25, 25))
        else:
            odo.move((100, -100))

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
