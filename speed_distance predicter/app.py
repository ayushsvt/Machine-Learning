from flask import Flask, render_template,request
from joblib import load
import numpy as np

def load_model():
     return load('speed_distance predicter/static/car_speed_dist.joblib')

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def index():
    if request.method=='POST':
        speed=int(request.form.get('speed'))
        inp=np.array([speed])
        inp = inp.reshape(-1,1)
        result=load_model().predict(inp)
        return render_template('index.html',result=result)
    return render_template('index.html')

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)
 