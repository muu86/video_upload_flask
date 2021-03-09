import os
import sys
import cv2
import json
import pickle
import numpy as np
from flask import Flask, request, send_from_directory, abort
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from anal_poses import anal

# openpose 패스 설정
op_path = "C:/openpose/bin/python/openpose/Release"
sys.path.append(op_path)
os.environ['PATH'] = os.environ['PATH'] + ';' + 'C:/openpose/bin'

# openpose import
try :
    import pyopenpose as op
except ImportError as e:
    raise e

# golfDB 패스 설정
golfdb_path = "C:\\golfdb\\"
sys.path.append(golfdb_path)

try:
    from test_video import SampleVideo, event_names
    from eval import ToTensor, Normalize
    from model import EventDetector
except ImportError as e:
    raise e

# 플라스크 시작
app = Flask(__name__, static_url_path='/static')


@app.route('/uploads', methods=['POST'])
def upload_file():
    print('성공')
    file = request.files['video']
    video_path = os.path.join(os.getcwd(), file.filename)
    file.save(video_path)

    """
    -----------------------
    골프 db 에서 모델을 가져온다
    -----------------------
    """
    print('golfdb 시작')
    ds = SampleVideo(video_path, transform=transforms.Compose([ToTensor(),
                                    Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True,
                              width_mult=1.,
                              lstm_layers=1,
                              lstm_hidden=256,
                              bidirectional=True,
                              dropout=False)

    save_dict = torch.load('models/swingnet_1800.pth.tar')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    print("Loaded model weights")

    seq_length = 64
    cwd = os.getcwd()
    print(cwd)
    save_path = cwd + '/static/output_images/'
    print(save_path)
    print('Testing...')
    for sample in dl:
        images = sample['images']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1

    events = np.argmax(probs, axis=0)[:-1]
    print('Predicted event frames: {}'.format(events))

    """
    cv2 비디오 캡쳐 오픈
    """
    cap = cv2.VideoCapture(video_path)

    confidence = []
    for i, e in enumerate(events):
        confidence.append(probs[e, i])
    print('Condifence: {}'.format([np.round(c, 3) for c in confidence]))

    """
        openpose 객체 오픈
        및 파라미터 설정
        """
    params = dict()
    params["model_folder"] = "C:\\openpose\\models\\"
    params["number_people_max"] = 1
    params["net_resolution"] = "-1x240"

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()

    # 각 프레임의 키포인트를 dict 로 모음
    key_data = dict()
    for i, e in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e)
        _, img = cap.read()
        # cv2.putText(img, '{:.3f}'.format(confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))

        # golfdb 가 뽑아낸 이미지를 op 객체에
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        key_data[i] = datum.poseKeypoints[0]
        cv2.imwrite(save_path + str(i) + '.png', datum.cvOutputData)

    cap.release()

    # 뽑아낸 키포인트를 분석
    swing_anal = anal.Anal(key_data)

    # check_all 함수에서 json.dumps를 리턴
    result = swing_anal.check_all()

    print(result)

    return result


@app.route('/get-images/<image_name><i>')
def get_images(image_name, i):

    image = send_from_directory("C:\\Users\\USER\\PycharmProjects\\flask_study\\static\\output_images",
                                filename=f"{i}.png", as_attachment=True)

    try:
        return image
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)