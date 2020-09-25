from flask import Flask,render_template,request
import pickle
import numpy as np
import math
app = Flask(__name__)
model = pickle.load(open('forest.pkl','rb'))


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict' , methods=['POST'])	
def predict():

	int_features = [int (x) for x in request.form.values()]
	final_features =[np.array(int_features)]
	prediction= model.predict_proba(final_features)
	output = '{0:.{1}f}'.format(prediction[0][1],2)
	

	if(output>str(0.5)):
		return render_template('index.html',pred='Your Forest is in Danger. \n probability of fire occuring is {}'.format(output) )
	else:
		return render_template('index.html',pred='Your Forest is in safe. \n probability of fire occuring is {}'.format(output) )



if __name__ == '__main__':
	app.run(debug=True)


