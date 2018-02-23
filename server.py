from flask import Flask, render_template, request, jsonify, redirect, url_for
from base64 import standard_b64decode
from png2jpg import png2jpg


import io
import os

app = Flask(__name__)
UPLOAD_FOLDER = '/Users/kenanyang/PycharmProjects/CombineP6/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
final = []
page = -1

import evaluation

prediction = evaluation.Evaluation()
import time
time.sleep(3)

@app.route("/", methods=['GET', 'POST'])
def index():
    global page
    global final
    page = page+1
    path = os.path.join(app.config['UPLOAD_FOLDER'], str(page))
    if not os.path.exists(path):
        os.mkdir(path)
        final.append('Running...')
    return redirect(url_for('usr_page', page=page))

@app.route("/usr_page/<int:page>", methods=['GET', 'POST'])
def usr_page(page):
    return render_template("CapstoneProject.html", page = page)


@app.route("/usr_page/upload/<int:page>", methods=['POST'])
def upload(page):
    base64 = request.form['hidden_data'].replace('data:image/png;base64,', '').replace(' ', '+')
    path = os.path.join(app.config['UPLOAD_FOLDER'], str(page)+'/test.png')
    file = io.open(path, 'wb')
    result = file.write(standard_b64decode(base64))
    file.close()
    png2jpg(page)
    global final

    final[page] = prediction.evaluate_one_image(page)
    return "The bytes written to test.png: " + str(result)

@app.route("/usr_page/updateResult/<int:page>", methods=['GET',  'POST'])
def updateResult(page):
    global final
    jsonStr = {'Result': final[page]}
    print(final)
    return jsonify(jsonStr)


if __name__ == "__main__":
    app.run(host='localhost', port=5001, debug=False)

#<script src="//ajax.googleapis.com/ajax/libs/jquery/1.8.2/jquery.min.js"></script>
# if __name__ == "__main__":
#         context = ('/private/etc/apache2/ssl/localhost.crt', '/private/etc/apache2/ssl/localhost.key')
#     app.run(host='localhost', port=5001,  ssl_context=context, threaded=True, debug=True)