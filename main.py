from __future__ import print_function
import sys
import os
from flask import Flask, flash, request, redirect, url_for,render_template,send_from_directory
import os

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
# from imageProcessing import allowed_file
from imageProcessing import image_inhance

UPLOAD_FOLDER = './userImage'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
def hello_world():
    # print(data, file=sys.stdout)
	return render_template('index.html')


@app.route('/result',methods=['GET', 'POST'])
def result():
    #   print('hello',file=sys.stdout)
	if request.method == 'POST':
		f = request.files['file']
		if f and allowed_file(f.filename):
			filename = secure_filename(f.filename)
			f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			processed_image= image_inhance(filename)
			if processed_image:
				print('hello')
				return redirect(url_for('uploaded_file', filename=processed_image))
			else: 
				return "unable to process image"
		
	return '''
				No file exit
			'''
        
@app.route('/show/<filename>')
def uploaded_file(filename):
    return render_template('index.html', filename=filename)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
   app.run(debug=True)