from flask import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')




app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'csv'}  # Allow only CSV files
app.secret_key = 'your_secret_key'  # Required for flash messages

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if the file part is present in the request
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        
        # If no file is selected, prompt for file selection
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        # If the file is a CSV, save it and load it into a DataFrame
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            df = pd.read_csv(filename)  # Read the CSV file into a DataFrame
            flash('Data uploaded successfully!')
            return render_template('upload.html', message='Data uploaded successfully!')
        else:
            flash('Only CSV files are allowed')
            return redirect(request.url)

    return render_template('upload.html')



@app.route('/view')
def view():
    global df, x_train, y_train, x_test, y_test
    df = pd.read_csv(r'uploads\Financial Distress.csv')
    # Assuming df is your DataFrame and 'financial_distress' is your target column
    df['Financial Distress'] = df['Financial Distress'].apply(lambda x: 0 if x > -0.50 else 1)

    ## SPlitting the data into Training and Testing
    x = df.drop('Financial Distress', axis = 1)
    y = df['Financial Distress']
    ## Balance the data
    sm = SMOTE()
    x, y = sm.fit_resample(x, y)
    ## Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    x_train = x_train[['x2', 'x3', 'x5', 'x8', 'x9', 'x10', 'x12', 'x13', 'x14', 'x16', 'x25',
       'x36', 'x42', 'x44', 'x46', 'x47', 'x48', 'x49', 'x52', 'x53', 'x61',
       'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x71',
       'x72', 'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 'x79', 'x81']]
    
    x_test = x_test[['x2', 'x3', 'x5', 'x8', 'x9', 'x10', 'x12', 'x13', 'x14', 'x16', 'x25',
       'x36', 'x42', 'x44', 'x46', 'x47', 'x48', 'x49', 'x52', 'x53', 'x61',
       'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x71',
       'x72', 'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 'x79', 'x81']]

    dummy = df.head(100)
    dummy = dummy.to_html()
    return render_template('view.html', data=dummy)


@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == "POST":
        model = request.form['Algorithm']

        if model == '1':
            gbr = GradientBoostingClassifier()
            gbr.fit(x_train, y_train)
            y_pred = gbr.predict(x_test)
            acc_gbr = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of Gradient Boosting Classifier = {acc_gbr}"
            return render_template('model.html', accuracy=msg)
        
        elif model == "2":
            adb = AdaBoostClassifier()
            adb.fit(x_train, y_train)
            y_pred = adb.predict(x_test)
            acc_adb = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of AdaBoost Classifier = {acc_adb}"
            return render_template('model.html', accuracy=msg)
        
        elif model == "3":
            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            acc_rf = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of Random Forest Classifier = {acc_rf}"
            return render_template('model.html', accuracy=msg)
        
        elif model == "4":
            # Initialize individual models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            lr = LogisticRegression(max_iter=1000, random_state=42)

            # If you want to use soft voting (probabilistic)
            VTC = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr) ], voting='soft')  # Use 'soft' for averaging predicted probabilities
            # Train the ensemble model with soft voting
            VTC.fit(x_train, y_train)
            y_pred = VTC.predict(x_test)
            acc_gnb = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of Voting Classifier = {acc_gnb}"
            return render_template('model.html', accuracy=msg)
        
        elif model == "5":
            # Define base classifiers
            base_classifiers = [ ('logistic', LogisticRegression(max_iter = 10000)), ('decision_tree', DecisionTreeClassifier()), ('random_forest', RandomForestClassifier())  ]
            # Define meta-classifier
            meta_classifier = LogisticRegression(max_iter = 10000)
            # Define the stacking classifier
            stc = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier )
            # Train the stacking classifier
            stc.fit(x_train, y_train)
            y_pred = stc.predict(x_test)
            acc_stc = accuracy_score(y_test, y_pred) * 100
            msg = f"Accuracy of Stacking Classifier = {acc_stc}"
            return render_template('model.html', accuracy=msg)        
    return render_template('model.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':

        f1 = float(request.form['x2'])
        f2 = float(request.form['x3'])
        f3 = float(request.form['x5'])
        f4 = float(request.form['x8'])
        f5 = float(request.form['x9'])
        f6 = float(request.form['x10'])
        f7 = float(request.form['x12'])
        f8 = float(request.form['x13'])
        f9 = float(request.form['x14'])
        f10 = float(request.form['x16'])
        f11 = float(request.form['x25'])
        f12 = float(request.form['x36'])
        f13 = float(request.form['x42'])
        f14 = float(request.form['x44'])
        f15 = float(request.form['x46'])
        f16 = float(request.form['x47'])
        f17 = float(request.form['x48'])
        f18 = float(request.form['x49'])
        f19 = float(request.form['x52'])
        f20 = float(request.form['x53'])
        f21 = float(request.form['x61'])
        f22 = float(request.form['x62'])
        f23 = float(request.form['x63'])
        f24 = float(request.form['x64'])
        f25 = float(request.form['x65'])
        f26 = float(request.form['x66'])
        f27 = float(request.form['x67'])
        f28 = float(request.form['x68'])
        f29 = float(request.form['x69'])
        f30 = float(request.form['x70'])
        f31 = float(request.form['x71'])
        f32 = float(request.form['x72'])
        f33 = float(request.form['x73'])
        f34 = float(request.form['x74'])
        f35 = float(request.form['x75'])
        f36 = float(request.form['x76'])
        f37 = float(request.form['x77'])
        f38 = float(request.form['x78'])
        f39 = float(request.form['x79'])
        f40 = float(request.form['x81'])

        lee = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40]]

        # Initialize individual models
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)

        # If you want to use soft voting (probabilistic)
        VTC = VotingClassifier(estimators=[ ('rf', rf), ('gb', gb), ('lr', lr) ], voting='soft')  # Use 'soft' for averaging predicted probabilities

        # Train the ensemble model with soft voting
        VTC.fit(x_train, y_train)
        result = VTC.predict(lee)
        print(result)

        if result == 0 :
            msg = f" The Company is financially healthy "
            return render_template('prediction.html', prediction = msg)
        else :
            msg = f" The Company is financially distressed  "
            return render_template('prediction.html', prediction = msg)
    return render_template('prediction.html')


if __name__ == '__main__':
    app.run(debug=True)