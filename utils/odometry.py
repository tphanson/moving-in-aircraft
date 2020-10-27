import numpy as np
import random
import math
import time
import cv2 as cv

BOT_WIDTH = 450
BOT_CENTER = [100, -100]

CAMERA_MATRIX = np.array([[0.4315, 0, 0.4695], [0, 0.7674, 0.5099], [0, 0, 1]])
RVEC = np.array([-np.deg2rad(11), 0, 0])
TVEC = np.array([0, 0, 1192.01])
DISTORTION_COEFF = np.array(
    [0.7365, 0.2228, 0, 0.0002, 0.0074, 1.097, 0.4153, 0.0477])


class Odometry:
    def __init__(self, botshell, image_shape, num_of_samples=100):
        self.botshell = botshell
        self.image_shape = image_shape
        self.num_of_samples = num_of_samples

    def _distort(self, pts, image_size):
        (h, w) = image_size
        pts, _ = cv.projectPoints(
            pts, RVEC, TVEC, CAMERA_MATRIX, DISTORTION_COEFF)
        pts[..., 0] = pts[..., 0]*w
        pts[..., 1] = pts[..., 1]*h
        return pts.squeeze().astype(np.int32)

    def _generate_ring(self, R, center, rad=math.pi):
        stop = int(R*math.cos(math.pi-rad))
        x = np.linspace(-R, stop, self.num_of_samples)
        y = -np.sqrt(np.square(np.full(self.num_of_samples, R)) - np.square(x))
        x = x + center[0]
        y = y + center[1]
        z = np.zeros(self.num_of_samples)
        return np.stack((x, y, z), axis=-1)

    def generate_driving_zone(self, R, rad):
        center = np.array([R, 0]) + BOT_CENTER
        outer_pts = self._generate_ring(
            R + np.sign(R)*BOT_WIDTH/2, center, rad)
        outer_pts = self._distort(outer_pts, self.image_shape)
        inner_pts = self._generate_ring(
            R - np.sign(R)*BOT_WIDTH/2, center, rad)
        inner_pts = self._distort(inner_pts, self.image_shape)
        polygon = np.append(inner_pts, np.flip(outer_pts, axis=0), axis=0)
        return polygon

    def _move_cmd(self, signal):
        (vleft, vfwd) = signal
        return f'set_velocity {vleft} {vfwd}\n'.encode()

    def get_velocity(self):
        self.botshell.sendall(b'get_velocity\n')
        vleft, vright = 0, 0
        try:
            data = self.botshell.recv(1024)
            [lvel, angvel] = data.decode('utf8').split(',')
            lvel, angvel = float(lvel), float(angvel)
            vleft = np.abs(800 * lvel + 450 * angvel)
            vright = np.abs(800 * lvel - 450 * angvel)
            if angvel <= 0:
                vleft, vright = 0, 0
        except ValueError:
            pass
        return vleft, vright

    def stop(self):
        cmd = self._move_cmd((0, 0))
        self.botshell.sendall(cmd)

    def avoid_obstacles(self):
        signal = random.choice([(-0.5, -0.5), (0.5, -0.5)])
        cmd = self._move_cmd(signal)
        counter = 0
        while counter < 5:  # Turn left in one second
            counter += 1
            self.botshell.sendall(cmd)
            time.sleep(0.05)

    def run_backward(self):
        cmd = self._move_cmd((0, -0.7))
        self.botshell.sendall(cmd)

    def run_forward(self):
        cmd = self._move_cmd((0, 0.7))
        self.botshell.sendall(cmd)
