from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)


model=joblib.load("emoji_predictor.joblib")

@app.route('/', methods=['GET'])
def form():
    # Render the form template
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form
    prediction =model.predict(pd.DataFrame([data]))
    
    return render_template("form.html", prediction=prediction)
    ## Extract age and gender
    #age = int(data['age'])
    #gender = int(data['gender'])
#
    ## Predict emoji
    #prediction = model.predict([[age, gender]])[0]
#
    ## Render the template with the prediction
    #return render_template('form.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
