from flask import Flask, render_template,request, jsonify
import random
import talk2car
import os
import json

app=Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/') 
def index(): 
        return render_template('index.html') 


@app.route('/show/', methods=['POST']) 
def show():
        num=int(random.randint(1,8000))
        talk2car.image(num)
        data={'number':num}
        with open("number.json", "w") as write_file:
                json.dump(data, write_file)

        return data




@app.route('/predict/', methods=['POST']) 
def predict():
        command = str(request.form.get('command'))
        with open("number.json") as write_file:
                data = json.load(write_file)
                number = int(data['number'])
        talk2car.predict(command,number)
        data={}
        return data





@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r 

if __name__ == '__main__': 
	app.run(debug=True) 



