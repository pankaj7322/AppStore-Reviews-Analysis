# Django Sentiment Analysis Project

## Overview
This project is a web application built using Django, allowing users to upload text data and analyze sentiment using two different models: BERT and TextBlob (NLP). The application provides features for user registration, login, file upload, sentiment analysis, and sentiment distribution visualization.

## Features
- **User Authentication:** Users can register, log in, and log out.
- **Text Preprocessing:** Uploaded text is cleaned using various techniques like lowercasing, punctuation removal, tokenization, and lemmatization.
- **Sentiment Analysis:** Sentiment is analyzed using two models:
  - **BERT Model:** Uses a pre-trained BERT model for sentiment analysis.
  - **TextBlob:** Uses the TextBlob library for a simpler sentiment analysis.
- **Visualization:** Sentiment distributions are shown using bar charts.

## Setup Instructions

### Prerequisites
Make sure you have the following installed:
- Python 3.7 or higher
- Django 3.x or higher
- PyTorch
- transformers
- TextBlob
- NLTK
- Matplotlib
- Seaborn

### Installation
1. **Clone the repository:**
    ```bash
    git clone [(https://github.com/pankaj7322/AppStore-Reviews-Analysis.git)](https://github.com/pankaj7322/AppStore-Reviews-Analysis.git)
    cd <cd AppStore-Reviews-Analysis>
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download necessary NLTK datasets:**
    Open a Python shell and run:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

5. **Run migrations:**
    ```bash
    python manage.py migrate
    ```

6. **Run the development server:**
    ```bash
    python manage.py runserver
    ```

7. Visit `http://127.0.0.1:8000/` in your browser.

## Application Flow

### Home Page
The home page is a simple entry point for the user.

### Register
Users can create an account by providing a username and password. The application checks for password confirmation and existing usernames.

### Login
Users can log in with their username and password. Upon successful login, they are redirected to the home page.

### File Upload
Users can upload a CSV file containing text data. The text is preprocessed, and sentiment predictions are performed.

### Sentiment Analysis
The uploaded text is analyzed using:
- **BERT Model:** Sentiment predictions are made using a pre-trained BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`).
- **TextBlob:** Sentiment is classified as positive, neutral, or negative based on polarity score.

### Sentiment Distribution
The sentiment distribution is visualized using a bar chart showing the counts of positive, negative, and neutral sentiments.

### Predictions
Users can input text manually to get a sentiment prediction.

## Models
### BERT Model
The BERT model used in this project is `nlptown/bert-base-multilingual-uncased-sentiment`, which is fine-tuned for sentiment analysis.

### TextBlob Model
TextBlob is used as a simpler, rule-based NLP model for sentiment analysis. It returns a sentiment polarity ranging from -1 (negative) to 1 (positive), which is then categorized as negative, neutral, or positive.

## Error Handling
- Errors are handled gracefully with error messages displayed to the user.
- Session data is cleared during user logout.

## Files and Directories
- `views.py`: Contains logic for handling requests, including sentiment analysis, text preprocessing, and file upload.
- `templates/`: Directory containing HTML templates for rendering the views (e.g., `index.html`, `register.html`, `login.html`).
- `static/`: Directory for static files such as CSS and JavaScript.

## Contributing
If you'd like to contribute to this project, please fork the repository, make your changes, and submit a pull request. All contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
