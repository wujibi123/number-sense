from flask import *

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('form.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    return render_template('display.html', color=request.form['color'])

app.run(debug=True)