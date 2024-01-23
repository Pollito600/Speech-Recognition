import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to extract features from an audio file
def extract_features(file_path, target_length=400):
    # Load audio data using librosa
    audio_data, _ = librosa.load(file_path, sr=None)
    # Perform Discrete Fourier Transform (DFT) on the audio data
    dft_result = np.fft.fft(audio_data)
    # Calculate the magnitude of the DFT result
    magnitude = np.abs(dft_result)
    # Adjust the length of the magnitude array to the target length
    if len(magnitude) < target_length:
        magnitude = np.pad(magnitude, (0, target_length - len(magnitude)))
    elif len(magnitude) > target_length:
        magnitude = magnitude[:target_length]
    return magnitude

# Function to plot Power Spectral Density (PSD) for a specific digit
def plot_psd_for_digit(features, digit, target_length=400):
    # Select a sample from the features array based on the specified digit
    sample = features[y_train == digit][0]
    # Calculate PSD and frequencies
    psd = np.abs(np.fft.fft(sample))**2
    freqs = np.fft.fftfreq(target_length, 1 / len(sample))
    # Plot the PSD
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:target_length // 2], 10 * np.log10(psd[:target_length // 2]), label=f'Digit {digit}')
    plt.title(f'Power Spectral Density (PSD) for Digit {digit}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.show()

# Function to plot PSD for a test file with actual and predicted digits
def plot_psd_for_test_file_with_actual_and_prediction(test_file, classifier, training_features, training_labels, target_length=400):
    # Extract features from the test file
    feature = extract_features(test_file)
    # Calculate PSD and frequencies for actual digit
    actual_psd = np.abs(np.fft.fft(feature))**2
    freqs = np.fft.fftfreq(target_length, 1 / len(feature))
    # Make a prediction using the classifier
    prediction = classifier.predict([feature])[0]
    # Select a training sample based on the predicted digit
    training_sample = training_features[training_labels == prediction][0]
    # Calculate PSD for the predicted digit
    training_psd = np.abs(np.fft.fft(training_sample))**2
    # Plot the PSD for actual and predicted digits
    plt.figure(figsize=(10, 6))
    plt.plot(freqs[:target_length // 2], 10 * np.log10(actual_psd[:target_length // 2]), label=f'Actual Digit ({os.path.basename(test_file)})', color='blue')
    plt.plot(freqs[:target_length // 2], 10 * np.log10(training_psd[:target_length // 2]), label=f'Predicted Digit {prediction}', color='green', linestyle='dashed')
    plt.title(f'Power Spectral Density (PSD) for Test File with Actual and Prediction')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.show()

# Function to calculate the ratio of low to high DFT coefficients
def calculate_ratio(feature):
    threshold = 100
    low_coeffs = np.sum(feature[:threshold])
    high_coeffs = np.sum(feature[threshold:])
    ratio = low_coeffs / high_coeffs if high_coeffs != 0 else 0
    return ratio

# Function to plot histogram of occurrences vs feature
def plot_occurrences_vs_feature_histogram(features, labels, feature_function, xlabel, ylabel, title):
    plt.figure(figsize=(15, 10))
    unique_digits = np.unique(labels)
    # Plot histograms for each digit
    for digit in unique_digits:
        digit_features = features[labels == digit]
        values = [feature_function(feature) for feature in digit_features]
        plt.hist(values, bins=20, alpha=0.5, label=f'Digit {digit}')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

# Function to load data from a folder
def load_data(folder_path):
    features = []
    labels = []
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            # Extract digit and features from the file
            digit = int(filename[0])
            feature = extract_features(file_path)
            features.append(feature)
            labels.append(digit)
    return np.array(features), np.array(labels)

# Define the path to the training folder
train_folder = "/Users/DFT_Project/Project A Data"
# Load training data and split into training and validation sets
X_train, y_train = load_data(train_folder)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Create and train a linear SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Plot PSD for each digit in the training set
for digit in np.unique(y_train):
    plot_psd_for_digit(X_train, digit)

# Plot histogram of occurrences vs the ratio of low to high DFT coefficients
plot_occurrences_vs_feature_histogram(X_train, y_train, calculate_ratio, 'Ratio of Low to High DFT Coefficients', 'Number of Occurrences', 'Histogram of Ratio of Low to High DFT Coefficients')

# Evaluate the classifier on the validation set
y_pred = classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Function to predict digits for a list of test files
def predict_digits(test_files, classifier):
    predictions = {}
    for test_file in test_files:
        feature = extract_features(test_file)
        prediction = classifier.predict([feature])
        predictions[test_file] = prediction[0]
    return predictions

# Define the path to the test folder
test_folder = "/Users/DFT_Project/Test Data"
# Create a list of test file paths
test_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder) if file.endswith(".wav")]
# Make predictions for each test file
all_predictions = predict_digits(test_files, classifier)

# Plot PSD for each test file with actual and predicted digits
for test_file in test_files:
    plot_psd_for_test_file_with_actual_and_prediction(test_file, classifier, X_train, y_train)

# Print predictions for each test file
for test_file, prediction in all_predictions.items():
    print(f"Prediction for {test_file}: Digit {prediction}")
