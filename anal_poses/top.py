import numpy as np
from utils import p3_angle


class Top:
    def __init__(self, kp):
        self.kp = kp
        self.feedback = dict()

    # 3번 Top 자세 분석
    # 왼 팔의 구부러짐 체크
    # 프로 스윙: 75~ 150 다수 / 150~ 175 / ~75
    def bending_left_arm(self):
        lshoulder = self.keypoints[3][5]
        lelbow = self.keypoints[3][6]
        lwrist = self.keypoints[3][7]

        angle = p3_angle(lshoulder, lelbow, lwrist)

        if angle >= 150.0:
            self.result[3] = {0: "good"}
        elif 100.0 <= angle <= 150.0:
            self.result[3] = {0: "normal"}
        else:
            self.result[3] = {0: "bad"}