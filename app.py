from flask import Flask, request, render_template  # Flask web framework components
from src.pipeline.predict_pipeline import CustomData, predict  # Custom prediction pipeline components

app = Flask(__name__)  # Initialize Flask application


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle prediction requests through web interface.

    GET: Returns the form page
    POST: Processes form data and returns prediction result
    """
    if request.method == 'GET':
        # Serve the form page for GET requests
        return render_template('index.html')
    else:
        # Create CustomData object from form inputs
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),  # Convert to integer
            writing_score=int(request.form.get('writing_score'))  # Convert to integer
        )

        # Convert input data to DataFrame format expected by model
        df = data.get_data_as_data_frame()

        # Get prediction from ML model pipeline
        results = predict(df)  # Assuming predict() returns an array-like result

        # Render template with prediction results
        return render_template('index.html', results=results[0])


if __name__ == "__main__":
    # Run the application on local development server
    app.run(host="127.0.0.1")  # Default port 5000
