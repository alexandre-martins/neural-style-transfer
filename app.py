import os
from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from style_transfer import NeuralTransfer

app = Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def response():
	# Getting content and style images proper names and filepaths
	content_img = request.files.get('content')
	style_img = request.files.get('style')
	content_filename = secure_filename(content_img.filename)
	style_filename = secure_filename(style_img.filename)

	content_filepath = os.path.join('./images/', content_filename)
	style_filepath = os.path.join('./images/', style_filename)

	# Defining output path
	output_path = './outputs/style_transfer_out.jpg'

	model = NeuralTransfer(content=content_filepath, style=style_filepath)
	model.train(epochs=5)

	return send_file(output_path, mimetype='image/jpg')

if __name__ == '__main__':
	app.run(host='0.0.0.0')