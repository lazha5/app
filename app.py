from flask import Flask
import pickle 
import  requests
from flask import (Flask,
                   request,
                   jsonify,
                   render_template)
import tensorflow as tf
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras import utils

standart_scaler =pickle.load(open('C:/Users/Admin/app/models/StandardScaler.pkl', 'rb'))
model_new = tf.keras.models.load_model('C:/Users/Admin/app/models/masha-classifier.h5')

app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
def main():
    if request.method == 'GET':
        return render_template('main.html')
    
    if request.method == 'POST':
        #get_data
        #poluchaem dann s form
        imt01	= float(request.form['imt'])
        krt02	= float(request.form['krt'])
        schph03	= float(request.form['schph'])
        calc04	= float(request.form['calc'])
        ph05= float(request.form['ph'])
        tsts06	= float(request.form['tsts'])
        kfk07= int(request.form['kfk'])
        vd308= float(request.form['vd3'])
        ptg09= float(request.form['ptg'])
        alf10	= float(request.form['alf'])
        bet11	= float(request.form['bet'])
        kzb12	= float(request.form['kzb'])
        kzl13 = float(request.form['kzl'])
        uzi14 = float(request.form['uzi'])
        
#preprocessing
##num
        X_nums_from_form =[imt01, 
                               krt02, 
                               schph03, 
                               calc04,
                               ph05,
                               tsts06,
			       kfk07,
			       vd308,
			       ptg09,
			       alf10,
			       bet11,
			       kzb12,
			       kzl13,
                               uzi14]
        print('X_nums', X_nums_from_form)

        #scaler
        X_scaled = standart_scaler.transform([X_nums_from_form])
        print('X_scaled:', X_scaled)

        #predict
        prediction = model_new.predict(X_scaled, verbose=0)
        print('Otvet:')
        print(prediction.round().argmax())

#result
        result = prediction.round().argmax()
                
        return render_template('predict.html', result = result)


if __name__ == '__main__':
  app.run(debug=True)