# Fake News Detection App

This repository contains a simple Streamlit application for detecting fake news using multiple machine learning algorithms. The app takes news content as input, processes it, and predicts whether it is fake or not using selected machine learning models.

## Features

- Supports multiple classifiers:
  - Logistic Regression
  - Decision Tree Classifier
  - Gradient Boosting Classifier
  - Random Forest Classifier
- Text preprocessing with custom functions for cleaning and vectorizing input.
- Interactive Streamlit UI for input and result display.

## Prerequisites

- Python 3.8 or higher

## Installation

1. Clone the repository:

   ```bash
   git <repository url>
   cd fake-news-detection
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## File Structure

- `Fake.csv` - Dataset of fake news samples
- `True.csv` - Dataset of true news samples
- `app.py` - Main application file
- `requirements.txt` - Python dependencies

## Usage

1. Upload `Fake.csv` and `True.csv` in the root directory.
2. Start the Streamlit app.
3. Enter the news content in the provided text area.
4. Select a classifier model from the sidebar.
5. Click **Classify** to get predictions.

## Dataset

The application uses two datasets:

- `Fake.csv`: Contains labeled fake news samples.
- `True.csv`: Contains labeled true news samples.

Each dataset should include the following columns:

- `title` (optional)
- `subject` (optional)
- `date` (optional)
- `text` (required): Contains the news content.

The app preprocesses the `text` column to remove unwanted characters, URLs, and punctuation.
