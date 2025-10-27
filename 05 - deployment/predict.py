# implementing a web service that will use our model to return predictions
import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

app = Flask('churn') # give an identity to your web service

@app.route('/predict', methods=['POST']) # use decorator to add Flask's functionality to our function
def predict():
    customer = request.get_json()
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn_decision = y_pred >= 0.5

    result = {
        'churn_probability': y_pred,
        'churn_decision' : bool(churn_decision)
    }
    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) # run the code in local machine with the debugging mode true and port 9696



