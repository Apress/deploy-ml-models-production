import pandas as pd
import numpy as np
import joblib
import streamlit

#load the model
model=open("linear_regression_model.pkl","rb")
lr_model=joblib.load(model)


def lr_prediction(var_1,var_2,var_3,var_4,var_5):
    pred_arr=np.array([var_1,var_2,var_3,var_4,var_5])
    preds=pred_arr.reshape(1,-1)
    preds=preds.astype(int)
    model_prediction=lr_model.predict(preds)
    return model_prediction

def run():
    streamlit.title("Linear Regression Model")
    html_temp="""
    
    """
 
    streamlit.markdown(html_temp)
    var_1=streamlit.text_input("Variable 1")
    var_2=streamlit.text_input("Variable 2")
    var_3=streamlit.text_input("Variable 3")
    var_4=streamlit.text_input("Variable 4")
    var_5=streamlit.text_input("Variable 5")
    
    prediction=""
    
    if streamlit.button("Predict"):
        prediction=lr_prediction(var_1,var_2,var_3,var_4,var_5)
    streamlit.success("The prediction by Model is {}".format(prediction))
    
if __name__=='__main__':
    run()