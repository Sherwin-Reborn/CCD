#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask
from joblib import load


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
import joblib

# Flask style
@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        income= float(request.form.get("income"))
        age = float(request.form.get("age"))
        loan = float(request.form.get("loan"))
        LR =load("DefaultLR.jl")
        Cart=load("DefaultCart.jl")
        RF=load("DefaultRF.jl")
        XGB=load("DefaultXGB.jl")
        MLP=load("DefaultMLP.jl")
        LR_pred = LR.predict([[income,age,loan]])
        Cart_pred = Cart.predict([[income,age,loan]])
        RF_pred = RF.predict([[income,age,loan]])
        XGB_pred = XGB.predict([[income,age,loan]])
        MLP_pred = MLP.predict([[income,age,loan]])
        LR_pred = LR_pred[0]
        Cart_pred = Cart_pred[0]
        RF_pred = RF_pred[0]
        XGB_pred = XGB_pred[0]
        MLP_pred = MLP_pred[0]
        print(LR_pred)
        LR_reply = "Logistic Regression Model - The Credit Card default is: " + str(LR_pred) 
        Cart_reply = "Decision Tree Cart Model - The Credit Card default is: " + str(Cart_pred)
        RF_reply = "Decision Tree Random Forest Model - The Credit Card default is: " + str(RF_pred)
        XGB_reply = "Decision Tree XGBoost Model- The Credit Card default is: " + str(XGB_pred)
        MLP_reply = "Neural Network MLP Model- The Credit Card default is: " + str(MLP_pred)
        
        return(render_template("website.html", result = LR_reply, result2=Cart_reply,result3=RF_reply,result4=XGB_reply,result5=MLP_reply))
    else: 
        return(render_template("website.html", result = "Predict Credit Card Default"))
    


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




