from flask import Flask, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from jinja2 import Template
import sys
import os
import argparse

# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/?highlight=upload
# 플라스크 api 사이트에 있는 샘플

# UPLOAD_FOLDER = '/uploads'
# ALLOWED_EXTENSIONS = {'mp4', 'jpg'}
#
app = Flask(__name__, static_url_path='/static')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
#
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # if 'file' not in request.files:
        #     flash('No file part')
        #     return redirect(request.url)
        #
        print(request.files.keys())
        print(request.files['data'])

        file = request.files['data']
        print(dir(file))
        print(file.filename)

        file.save(os.path.join(os.getcwd(), 'uploads.mp4'))



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
            video_path = os.path.join(os.getcwd(), 'uploads.mp4')

            cap = cv2.VideoCapture(video_path)
            print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = os.path.join(os.getcwd(), 'static/output.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()

                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                out.write(datum.cvOutputData)

                print(datum.poseKeypoints)
                print(cap.get(cv2.CAP_PROP_POS_FRAMES))

            cap.release()
            print('end')

        except Exception as e:
            print(dir(e))
            print(e)
            sys.exit(-1)
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)

        # if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # return redirect(url_for('upload_file',
        #                         filename=filename))

        # return '''
        # <!doctype html>
        # <title>Upload new File</title>
        # <h1>Upload new File</h1>
        # <form method=post enctype=multipart/form-data>
        #     <input type=file name=file>
        #     <input type=submit value=Upload>
        # </form>
        # '''
        return "good"

    if request.method == 'GET':
        return '''
        <h1>mj</h1>
        '''


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
