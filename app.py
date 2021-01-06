from flask import Flask,render_template,url_for,request
#from sklearn.externals import joblib
import pandas as pd
import os
import pickle
import joblib
from webScraper import required_details


filename = 'newsPrediction_pickle.pkl'
clf = joblib.load(open(filename, 'rb'))

app = Flask(__name__)
@app.route('/')

def home():
	return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        news_data = required_details(message)
        data=pd.DataFrame(news_data)
        data=data.transpose()
        data.columns =[' n_tokens_title',' n_tokens_content',' n_unique_tokens',' n_non_stop_unique_tokens',
        ' num_imgs',' average_token_length',' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday',
        ' weekday_is_thursday',' weekday_is_friday',' weekday_is_saturday',' weekday_is_sunday',' is_weekend',
        ' global_subjectivity',' global_sentiment_polarity',' title_subjectivity',' title_sentiment_polarity']
        my_prediction = clf.predict(data)
        my_ans = int(my_prediction[0])
        if my_ans >= 20000:
            ans = 'Very High chances of getting Popular (Estimated views - '
        elif my_ans >= 10000:
            ans = 'High chances of getting Popular (Estimated views - '
        elif my_ans >= 5000:
            ans = 'Slight chances of getting Popular (Estimated views - '
        else:
            ans = 'Few chances of getting Popular (Estimated views - '
        ans = ans + str(my_ans) +')'
        return render_template('result.html',prediction = ans)


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(debug=True,host='0.0.0.0', port=port)
