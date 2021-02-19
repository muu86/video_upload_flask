import sys
import os
import argparse

try:
    op_path = "C:/openpose/bin/python/openpose/Release"
    try:
        sys.path.append(op_path)
        os.environ['PATH'] = os.environ['PATH'] + ';' + 'C:/openpose/bin'

        import pyopenpose as op

    except ImportError as e:
        raise e

    # cv2 import
    cv2_path = "C:/Users/USER/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/python-3.7"
    try:
        sys.path.append(cv2_path)
        sys.path.append('C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages')

        import cv2

    except ImportError as e:
        raise e

    params = dict()
    params["model_folder"] = "C:\\openpose\\models\\"
    params["number_people_max"] = 1

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()
    video_path = "C:/swing_data/updates/tiger_woods_2.mp4"
    cap = cv2.VideoCapture(video_path)
    print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()

        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        cv2.imshow('hi', datum.cvOutputData)
        cv2.waitKey(0)
        print(datum.poseKeypoints)
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cap.release()
    print('end')

except Exception as e:
    print(dir(e))
    print(e)
    sys.exit(-1)