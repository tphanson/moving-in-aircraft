import os
import datetime
import time
import cv2 as cv
from utils.ros import ROSImage

USB_PATH = '/storage/3115-7E1B/ds/'
DESTINATION = datetime.datetime.now().strftime(
    '%c').replace(' ', '_').replace(':', '_')


def calibrate():
    ros = ROSImage()
    talker = ros.gen_talker('/ds_calib/image/compressed')
    camera = cv.VideoCapture(1)
    step = 0
    count = 0
    while True:
        step += 1
        ok, img = camera.read()
        print("================== Image:", step, count, ok)
        if ok and step % 100 == 0:
            count += 1
            talker.push(img)
        time.sleep(0.1)  # Limit at 10Hz


def collect():
    # Create save folder
    os.mkdir(USB_PATH+DESTINATION)
    # Main process
    camera = cv.VideoCapture(1)
    step = 0
    count = 0
    while True:
        step += 1
        ok, img = camera.read()
        if ok and step % 100 == 0:
            count += 1
            name = USB_PATH + DESTINATION + '/' + str(count) + '.jpg'
            print("================== Image:", name)
            cv.imwrite(name, img)
        time.sleep(0.1)
