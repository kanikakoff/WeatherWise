import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
data = {
    'Temperature': [22, 25, 21, 30, 29, 27, 24, 23, 26, 28],
    'Humidity': [85, 90, 88, 70, 75, 80, 85, 90, 95, 77],
    'Precipitation': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    'Rain': [0, 1, 0, 0, 0, 1, 1, 0, 0, 1]  # Target variable
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Temperature', 'Humidity', 'Precipitation']]
y = df['Rain']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Function to predict rain and provide advice
def predict_rain_with_advice(temperature, humidity, precipitation):
    new_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Precipitation': [precipitation]
    })
    prediction = model.predict(new_data)
    
    if prediction[0] == 1:
        advice = "It's likely to rain. Consider carrying an umbrella and wearing waterproof clothing. Avoid outdoor activities if possible."
    else:
        advice = "It's not expected to rain. You can enjoy outdoor activities, but it’s always good to be prepared for unexpected weather changes."

    return 'Rainy' if prediction[0] == 1 else 'Not Rainy', advice

# Get user input
def get_user_input():
    try:
        temp = float(input("Enter the temperature (°C): "))
        hum = float(input("Enter the humidity (%): "))
        precip = float(input("Enter the precipitation (mm): "))
        return temp, hum, precip
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None

# Main program
if __name__ == "__main__":
    user_input = get_user_input()
    if user_input:
        temp, hum, precip = user_input
        result, advice = predict_rain_with_advice(temp, hum, precip)
        print(f'Prediction for Temperature={temp}, Humidity={hum}, Precipitation={precip}: {result}')
        print(f'Advice: {advice}')
