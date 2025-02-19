import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Define paths for saving models and uploads
UPLOAD_FOLDER = './uploads'
MODEL_FOLDER = './models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Ensure that the directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Utility function to train and save the model as .keras
def train_lstm_model(csv_file, epochs=50):
    data = pd.read_csv(csv_file)

    # Check if required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}")

    X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = data['Adj Close']

    # Normalize features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Reshape for LSTM input
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, random_state=42)

    # Define the LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Add early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=32, 
              validation_split=0.2, 
              callbacks=[early_stopping])

    # Save the model
    model_filename = os.path.join(MODEL_FOLDER, f"{os.path.basename(csv_file).split('.')[0]}.keras")
    model.save(model_filename)

    return model, scaler_X, scaler_y, model_filename

# Predict based on the first row of the CSV using the saved .keras model
def predict_adj_close(csv_file, model_filename, scaler_X, scaler_y):
    data = pd.read_csv(csv_file)

    # Ensure required columns exist
    if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
        raise ValueError("CSV file must contain the columns: Open, High, Low, Close, Volume")

    new_data = data.iloc[0][['Open', 'High', 'Low', 'Close', 'Volume']].values.reshape(1, -1)
    new_X_scaled = scaler_X.transform(new_data)
    new_X_reshaped = new_X_scaled.reshape(new_X_scaled.shape[0], 1, new_X_scaled.shape[1])

    # Load the model
    model = load_model(model_filename)

    # Make prediction
    prediction_scaled = model.predict(new_X_reshaped)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

    return prediction[0][0]

# Route for handling file uploads and predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                message = "No selected file"
            elif file and file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                try:
                    # Train the model on the uploaded CSV
                    model, scaler_X, scaler_y, model_filename = train_lstm_model(filepath)
                    message = "Model trained and saved successfully"
                except Exception as e:
                    message = f"Error: {str(e)}"

        # If a CSV file is selected from the dropdown for prediction
        if 'csv_file' in request.form:
            selected_file = request.form['csv_file']
            model_filename = os.path.join(MODEL_FOLDER, f"{selected_file.split('.')[0]}.keras")

            if os.path.exists(model_filename):
                try:
                    # Load scalers and predict
                    _, scaler_X, scaler_y, _ = train_lstm_model(os.path.join(UPLOAD_FOLDER, selected_file), epochs=1)
                    prediction = predict_adj_close(os.path.join(UPLOAD_FOLDER, selected_file), model_filename, scaler_X, scaler_y)
                except Exception as e:
                    message = f"Error: {str(e)}"

    # List all CSV files in the uploads folder
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]

    return render_template('index.html', csv_files=csv_files, prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
