import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from api import *
app = Flask(__name__)


cfg_file = 'configs/cocobox_r50_base.yaml'
cfg = load_cfg(cfg_file)
mode = 'TEST'
prepare_device(cfg, mode)
detector = prepare_detector(cfg, mode)
dataset = prepare_dataset(cfg, detector, mode)
inferencer = Inferencer(cfg, detector, dataset, mode)


ALLOWED_EXTENSIONS = set(['jpg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    upload_path = os.path.join('static/images', 'upload.jpg')
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return render_template('detx2d.html', upload_path=upload_path)
        user_input = request.form.get("name")
        basepath = os.path.dirname(__file__)
        secure_filename_ = secure_filename(f.filename)
        abs_path = os.path.join(basepath, 'static/images', secure_filename_)
        rel_path = os.path.join('static/images', secure_filename_)
        f.save(abs_path)

        # inference
        img = Image.open(rel_path)
        pred = inferencer.pred(img)
        name = 'static/images' + '/pred_' + secure_filename_.split('.')[0]+'.jpg'
        dataset.show(img, pred, name)

        return render_template('detx2d.html', upload_path=name)
    else:
        return render_template('detx2d.html', upload_path=upload_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


