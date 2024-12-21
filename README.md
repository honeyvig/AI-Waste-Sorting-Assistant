# AI-Waste-Sorting-Assistant
RecycleMate is an AI-powered mobile application designed to assist users in sorting waste correctly. By leveraging image recognition technology, the app determines if waste items are recyclable, compostable, or trash, and provides actionable recommendations. The app also offers location-based recycling information, weekly eco-challenges, and fosters environmental education.

Project Goals

Use AI-driven image recognition to identify waste items.
Provide location-based recycling center information.
Encourage eco-friendly habits through weekly challenges.
Foster environmental awareness and community impact.

System Workflow

1. Waste Scanning Process

Step 1: User opens camera from app.
Step 2: Image is captured and sent to the backend or processed locally.
Step 3: AI model classifies the image into categories (Recyclable, Compost, Trash).
Step 4: Result displayed to user with relevant sorting advice.

2. Recycling Center Location

Step 1: User taps "Find Recycling Center".
Step 2: App uses location data (GPS) and fetches nearby recycling centers.
Step 3: Filters are applied (e.g., plastic, glass).
Step 4: Directions and contact information provided.

3. Weekly Challenges

Step 1: User receives a new challenge.
Step 2: Logs activity (manual input or AI confirmation).
Step 3: Completion progress tracked.
Step 4: User earns rewards or points.

4. Community Dashboard

Step 1: Aggregate data of waste items sorted and carbon footprint impact.
Step 2: Display visual metrics (charts, statistics).
Step 3: Educational content pushed to users.

-------------------------
To implement the RecycleMate AI-powered mobile application, we can break it down into four main functional components and create Python code snippets to support each feature. Let's approach this step by step and include relevant functions for each workflow.
Key Components:

    Waste Scanning with Image Recognition: This involves image processing and classification to determine if an item is recyclable, compostable, or trash.
    Recycling Center Location: Using GPS and location-based services to recommend nearby recycling centers.
    Weekly Eco-Challenges: Allowing users to participate in eco-challenges and log activities.
    Community Dashboard: Aggregating environmental data and visualizing progress.

Python Code for the RecycleMate System

Let's write code snippets for each of these components.
1. Waste Scanning Process: AI Model Classification

This part requires an AI model trained on image recognition to classify waste items as recyclable, compostable, or trash.

We will use the TensorFlow and Keras libraries for image classification. The model can be trained with labeled data (images of recyclable, compostable, and trash items).

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained model (for demonstration purposes)
model = tf.keras.models.load_model('waste_classifier_model.h5')

# Function to classify the image
def classify_waste_item(image_path):
    # Load the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    
    # Predict the class of the image
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)
    
    # Map the class index to categories
    categories = ['Recyclable', 'Compost', 'Trash']
    return categories[class_index[0]]

# Example usage
image_path = 'waste_item.jpg'  # Path to an image
result = classify_waste_item(image_path)
print(f"The item is classified as: {result}")

2. Recycling Center Location

For the location-based feature, we can use geopy to fetch the user's current location and use it to find nearby recycling centers.

from geopy.geocoders import Nominatim
import requests

# Function to get current location using geopy
def get_current_location():
    geolocator = Nominatim(user_agent="recycleMateApp")
    location = geolocator.geocode("Your Address or City Name Here")
    return location.latitude, location.longitude

# Function to fetch nearby recycling centers
def find_nearby_recycling_centers(lat, lon, waste_type):
    # Example API endpoint (Assuming an API exists for recycling center data)
    api_url = f"https://api.recycling-centers.com/centers?lat={lat}&lon={lon}&type={waste_type}"
    
    response = requests.get(api_url)
    if response.status_code == 200:
        centers = response.json()
        return centers
    else:
        return "No recycling centers found."

# Example usage
lat, lon = get_current_location()
waste_type = 'plastic'  # Can be 'plastic', 'glass', etc.
centers = find_nearby_recycling_centers(lat, lon, waste_type)
print(centers)

3. Weekly Eco-Challenges

This functionality will include user registration, receiving challenges, logging progress, and tracking rewards.

class EcoChallenge:
    def __init__(self, challenge_name, points):
        self.challenge_name = challenge_name
        self.points = points
        self.completed = False
    
    def mark_completed(self):
        self.completed = True
        return f"Challenge '{self.challenge_name}' completed! You earned {self.points} points."

# Sample weekly challenges
challenges = [
    EcoChallenge("Reduce plastic usage for a week", 50),
    EcoChallenge("Compost organic waste for a month", 100),
    EcoChallenge("Collect 10 recyclables this week", 30)
]

# User progress tracking
user_points = 0

def log_challenge_progress(challenge_index):
    global user_points
    challenge = challenges[challenge_index]
    if not challenge.completed:
        user_points += challenge.points
        print(challenge.mark_completed())
    else:
        print(f"Challenge '{challenge.challenge_name}' already completed.")

# Example usage: User completes the second challenge
log_challenge_progress(1)
print(f"Total points earned: {user_points}")

4. Community Dashboard: Data Visualization

We can use Python's matplotlib and pandas to create a community dashboard that tracks the environmental impact, such as waste items sorted, carbon footprint saved, etc.

import matplotlib.pyplot as plt
import pandas as pd

# Sample data for visualization
data = {
    'User': ['Alice', 'Bob', 'Charlie', 'David'],
    'Recyclables': [120, 150, 100, 180],
    'Compostables': [50, 30, 80, 40],
    'Trash': [30, 60, 50, 20],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create bar chart for the waste items sorted
def plot_community_dashboard(df):
    df.set_index('User', inplace=True)
    df.plot(kind='bar', stacked=True)
    
    plt.title('Community Waste Sorting Progress')
    plt.xlabel('User')
    plt.ylabel('Items Sorted')
    plt.show()

# Display the community dashboard
plot_community_dashboard(df)

Full Workflow Integration:

These individual components (waste scanning, location-based recycling, weekly challenges, and community dashboard) are the backbone of the RecycleMate app. The app will combine all these features into an intuitive, user-friendly interface (mobile app).
System Workflow:

    Waste Scanning:
        The user opens the camera, takes a photo of the item, and the app classifies it using the AI model.
        Displays results with relevant sorting advice.

    Recycling Center Location:
        The app fetches the user's location and displays nearby recycling centers based on waste type.

    Weekly Challenges:
        Users receive new challenges, log their actions, and earn rewards or points for completing eco-friendly tasks.

    Community Dashboard:
        The app aggregates data on how many waste items have been sorted, visualizes it, and displays the community impact.

Conclusion:

This Python code demonstrates how to implement the key features of the RecycleMate application, including waste scanning, location-based services, eco-challenges, and a community dashboard. These can be further integrated with a mobile app framework such as React Native or Flutter to deliver an intuitive user experience on mobile devices.
