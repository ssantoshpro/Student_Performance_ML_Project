from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipleline
from logging import FileHandler,WARNING

# print a nice greeting.
def say_hello(username = "World"):
    return '<p>Hello %s!</p>\n' % username

# some bits of text for the page.
header_text = '''
    <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
    <p><em>Hint</em>: This is a RESTful web service! Append a username
    to the URL (for example: <code>/Thelonious</code>) to say hello to
    someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__,template_folder = 'templates')
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
# add a rule for the index page.
# application.add_url_rule('/', 'index', (lambda: header_text +
#     say_hello() + instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
application.add_url_rule('/<username>', 'hello', (lambda username:
    header_text + say_hello(username) + home_link + footer_text))


# application = Flask(__name__)#,static_folder=static_path)

# app = application 
# #testing

## Route for home page flow
@application.route("/")
def index():
    return render_template("index.html")


@application.route('/predictdata',methods=['GET','POST'])
def predict_dataPoint():
    if request.method=='GET':
        return render_template('home.html',results ="Please fill the dependent values above")
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
        return render_template('home.html',results = str(round(results[0],2)))

if __name__ == '__main__':
    application.run(host='0.0.0.0',port=5000,debug=True)
