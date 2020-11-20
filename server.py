import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import time
from api import *
app = Flask(__name__)

# TODO: change config file
cfg_file = 'configs/cocobox_r50_base.yaml'

# load model inferencer
cfg = load_cfg(cfg_file)
mode = 'TEST'
prepare_device(cfg, mode)
detector = prepare_detector(cfg, mode)
dataset = prepare_dataset(cfg, detector, mode)
inferencer = Inferencer(cfg, detector, dataset, mode)

# global
ALLOWED_EXTENSIONS = set(['jpg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# app
@app.route('/', methods=['POST', 'GET'])
def hello_world():
    upload_default = os.path.join('static/images', 'upload.jpg')
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return render_template('detx2d.html', upload_path=upload_default)
        user_input = request.form.get("name")
        basepath = os.path.dirname(__file__)

        # get path
        filename = secure_filename(f.filename)
        filename_sp = filename.split('.')
        filename = filename_sp[0] + str(int(time.time()*10)) + '.' + filename_sp[1]
        abs_path = os.path.join(basepath, 'static/images', filename)
        rel_path = os.path.join('static/images', filename)

        # save file
        f.save(abs_path)

        # inference
        img = Image.open(rel_path)
        pred = inferencer.pred(img)

        # save pred
        pred_rel_path = 'static/images' + '/pred_' + filename.split('.')[0]+'.jpg'
        dataset.show(img, pred, pred_rel_path)

        return render_template('detx2d.html', upload_path=pred_rel_path)
    else:
        return render_template('detx2d.html', upload_path=upload_default)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
