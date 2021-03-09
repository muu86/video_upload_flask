import numpy as np
# import address
import json
from anal_poses.backswing import Backswing
from anal_poses.utils import MyEncoder


class Anal:
    result = dict()

    def __init__(self, kp):
        self.kp = kp
        self.backswing = Backswing(kp[2])

    def check_all(self):
        self.result[2] = self.backswing.run()
        return json.dumps(self.result, cls=MyEncoder)

