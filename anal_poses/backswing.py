import numpy as np
from anal_poses.utils import p3_angle


# 2번 자세
class Backswing:
    def __init__(self, kp):
        self.kp = kp
        self.feedback = dict()

    # 스웨이 체크
    # 골반과 몸통이 회전하는 것이 아니라 오른쪽으로 밀리면서 체중 이동 하는 것
    def sway(self):
        lhip_address = self.kp[9]
        lhip_backswing = self.kp[9]

        diff = np.array(lhip_address) - np.array(lhip_backswing)

        # 프로는 -0.05 ~ 0.05 사이의 차이를 보임
        if -0.05 <= diff[0] <= 0.05:
            self.feedback["sway"] = {
                0: "good",
                1: diff[0]
            }
        elif -0.2 <= diff[0] <= 0.2:
            self.feedback["sway"] = {
                0: "normal",
                1: diff[0]
            }
        else:
            self.feedback["sway"] = {
                0: "bad",
                1: diff[0]
            }

    # 헤드 포지션 체크
    # 어드레스 시 코의 위치와 백스윙 시 코의 위치 체크
    # 좌우의 움직임보다 위 아래로의 움직임이 중요
    # y 축의 변화를 체크한다
    def head_position(self):
        nose_address = self.kp[0]
        nose_backswing = self.kp[0]
        diff = np.array(nose_address) - np.array(nose_backswing)

        if -1 <= diff[1] <= 1:
            self.feedback['head_position'] = {
                0: "good",
                1: diff[1]
            }
        elif -1.5 <= diff[1] <= 1.5:
            self.feedback['head_position'] = {
                0: "normal",
                1: diff[1]
            }
        else:
            self.feedback['head_position'] = {
                0: "bad",
                1: diff[1]
            }


    # 모든 함수를 실행시킴
    def run(self):
        self.sway()
        self.head_position()

        return self.feedback