import pandas as pd
import numpy as np
import sklearn
import joblib
from flask import Flask,render_template,request
app=Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])

def predict():
	if request.method =='POST':
		print(request.form.get('var_1'))
		print(request.form.get('var_2'))
		print(request.form.get('var_3'))
		print(request.form.get('var_4'))
		print(request.form.get('var_5'))
		try:
			var_1=float(request.form['var_1'])
			var_2=float(request.form['var_2'])
			var_3=float(request.form['var_3'])
			var_4=float(request.form['var_4'])
			var_5=float(request.form['var_5'])
			pred_args=[var_1,var_2,var_3,var_4,var_5]
			pred_arr=np.array(pred_args)
			print(pred_arr)
			preds=pred_arr.reshape(1,-1)
			model=open("linear_regression_model.pkl","rb")
			lr_model=joblib.load(model)
			model_prediction=lr_model.predict(preds)			
			model_prediction=round(float(model_prediction),2)
		except ValueError:
			return "Please Enter valid values"
	return render_template('predict.html',prediction=model_prediction)
if __name__=='__main__':
	app.run(host='0.0.0.0')
