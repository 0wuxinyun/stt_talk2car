from flask import Flask, render_template, request, jsonify 
import random
app = Flask(__name__) 
 
@app.route('/') 
def index(): 
	return render_template('index.html') 
 
@app.route('/random/', methods=['POST']) 
def square(): 
	num = int(request.form.get('number', 0)) 
	rl=[]
	for i in range(num):
		rl.append(random.randint(0,9))
	data = {'random': rl} 
	data = jsonify(data) 
	return data 
 
if __name__ == '__main__': 
	app.run(debug=True) 
