from flask import Flask, render_template, request, url_for

from model import *


app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def home():
    data = pd.read_csv('UCI_Credit_Card.csv')
    target_col = "Default"
    features = [x for x in data.columns if x != target_col]
    if request.method == 'POST':
        user_input_sampling = request.form.get("resampling")
        user_input_features = request.form.getlist("features")
        user_input_scaling  = request.form.get("scaling")
        user_input_model  = request.form.get("model")
    else: 
        user_input_sampling = 'Oversampling' #request.args.get("resampling")
        user_input_features = ['PAY_0', 'BILL_AMT1','PAY_AMT2'] #request.args.getlist("features")
        user_input_scaling  = 'Standard' #request.args.get("scaling")
        user_input_model  = 'random forest' #request.args.get("model")
    results = model_pipeline(data, target_col , user_input_features , user_input_sampling, user_input_scaling, user_input_model)
    return render_template('index.html', results=results, features = features, accuracy = round(results['metrics']['accuracy'] * 100,2), precision = round(results['metrics']['precision'] * 100,2), recall = round(results['metrics']['recall'] * 100, 2) , f1 = round(results['metrics']['f1'] * 100, 2) , roc_auc = round(results['metrics']['roc_auc']*100,2) )

    

if __name__ == '__main__':
    app.run(debug=True)


