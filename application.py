import os
import sys
# sys.path.append("/path_to_your_app/eb-flask")
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipleline

#static_path = os.path.abspath(r"templates\assets")
application = app = Flask(__name__)#,static_folder=static_path)


## Route for home page
@application.route("/")
def index():
    return render_template("index.html")

@application.route("/components.html")
def components():
    return render_template("components.html")

@application.route('/predictdata',methods=['GET','POST'])
def predict_dataPoint():
    if request.method=='GET':
        return render_template('home.html')
    else :
        data = CustomData(
            gender = request.form.get("gender"),
            race_ethnicity = request.form.get("race_ethnicity"),
            parental_level_of_education = request.form.get("parental_level_of_education"),
            lunch = request.form.get("lunch"),
            test_preparation_course = request.form.get("test_preparation_course"),
            math_score = request.form.get("math_score"),
            reading_score = request.form.get("reading_score"),
            writing_score = request.form.get("writing_score"),
        )
        pred_df = data.get_data_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipleline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results = results[0])

if __name__=="__main__":
    application.run('localhost',5000,debug=True)
