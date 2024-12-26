from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
model_filename = 'model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    item_identifier = request.form['item_identifier']
    item_weight = float(request.form['item_weight'])
    item_fat_content = request.form['item_fat_content']
    item_visibility = float(request.form['item_visibility'])
    item_type = request.form['item_type']
    item_mrp = float(request.form['item_mrp'])
    outlet_identifier = request.form['outlet_identifier']
    outlet_establishment_year = int(request.form['outlet_establishment_year'])
    outlet_size = request.form['outlet_size']
    outlet_location_type = request.form['outlet_location_type']
    outlet_type = request.form['outlet_type']

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Item_Identifier': [item_identifier],
        'Item_Weight': [item_weight],
        'Item_Fat_Content': [item_fat_content],
        'Item_Visibility': [item_visibility],
        'Item_Type': [item_type],
        'Item_MRP': [item_mrp],
        'Outlet_Identifier': [outlet_identifier],
        'Outlet_Establishment_Year': [outlet_establishment_year],
        'Outlet_Size': [outlet_size],
        'Outlet_Location_Type': [outlet_location_type],
        'Outlet_Type': [outlet_type],
    })

    # Encode categorical variables (same as during training)
    encoder = LabelEncoder()
    for col in ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 
                'Outlet_Identifier', 'Outlet_Size', 
                'Outlet_Location_Type', 'Outlet_Type']:
        input_data[col] = encoder.fit_transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)[0]  # Get the predicted sales value

    # Generate insights based on prediction ranges
    if prediction < 1000:
        insight = "This product is not selling well."
    elif 1000 <= prediction <= 3000:
        insight = "This product is selling at an average rate."
    else:
        insight = "This product is selling well."

    # Add additional insights based on store characteristics
    additional_insights = []
    if outlet_size == 'Large' and prediction > 3000:
        additional_insights.append("This product sells well in big stores.")
    if outlet_location_type == 'Tier 1' and outlet_type.startswith('Supermarket'):
        additional_insights.append("This product sells more on weekends.")

    # Render the template with prediction and insights
    return render_template(
        'index.html', 
        prediction=round(prediction, 2), 
        insight=insight, 
        additional_insights=additional_insights
    )

if __name__ == '__main__':
    app.run(debug=True)
