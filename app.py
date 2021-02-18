import os
from flask import Flask, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from jinja2 import Template


# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/?highlight=upload
# 플라스크 api 사이트에 있는 샘플

# UPLOAD_FOLDER = '/uploads'
# ALLOWED_EXTENSIONS = {'mp4', 'jpg'}
#
app = Flask(__name__)
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
        return 'good'

    if request.method == 'GET':
        return '''
        <h1>mj</h1>
        '''


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
