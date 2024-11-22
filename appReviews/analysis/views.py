from django.shortcuts import render,redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.models import User
import pandas as pd
from io import StringIO
import nltk
import string
import re
import matplotlib
matplotlib.use('Agg')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification,BertTokenizer, BertForSequenceClassification
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment").to(device)

def home(request):
    return render(request, 'index.html')



def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        confirm_password = request.POST['password1']

        if password != confirm_password:
            messages.error(request, "Password do not match")
            return render(request, 'register.html', {'error': 'Password do not match'})
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return render(request, 'register.html', {'error': "Username already exists"})

        user = User.objects.create_user(username=username, password=password)
        user.save()

        # login(request, user)

        return render(request, 'register.html', {'error': "User Registered Successfully"})
    else:
        return render(request, 'register.html')

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            return redirect('home')  # Redirect to home after login
        else:
            # Handle invalid login
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    return render(request, 'login.html')

def preprocess_text_v3(text):
    # Ensure the text is a string (this prevents errors if there are null or non-string values)
    if not isinstance(text, str):
        return ""

    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove punctuation (except for hyphens and apostrophes)
    text = ''.join([char for char in text if char not in string.punctuation or char in ['-', '\'']])

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove emojis using a regular expression
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # This removes non-ASCII characters (including emojis)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens (to get the root form of the word)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Optionally, remove any remaining short words (e.g., length <= 2)
    tokens = [word for word in tokens if len(word) > 2]

    return " ".join(tokens)

def logout_user(request):
    try:
        # Clear the session data (remove all session data)
        request.session.flush()

        # Optionally, add a success message
        messages.success(request, "You have been logged out successfully.")

        # Redirect to a login page or home page after logout
        return redirect('login')  # Update this to your login page or home page URL

    except Exception as e:
        messages.error(request, f"Error logging out: {str(e)}")
        return redirect('home')  # Redirect to the home page if there's an error

def upload_file(request):
    if request.method == "POST" and request.FILES.get('file-upload'):
        uploaded_file = request.FILES['file-upload']

        # Read the uploaded file directly into a pandas DataFrame (without saving to disk)
        try:
            # Read the uploaded CSV file into pandas DataFrame
            file_content = uploaded_file.read().decode('utf-8')  # Decode the byte content to string
            data = StringIO(file_content)  # StringIO lets pandas treat the string as a file object
            df = pd.read_csv(data)  # Read CSV data into a DataFrame
            df['cleaned_review'] = df['text'].apply(preprocess_text_v3)

            # Convert DataFrame to JSON to store in session
            df_json = df.to_json(orient='split')  # You can also use 'records' or 'columns'

            # Store the DataFrame in session (as a string)
            request.session['df_data'] = df_json

            request.session['file_name'] = uploaded_file.name
            request.session['num_rows'], request.session['num_cols'] = df.shape 

            # Get the top 5 rows of the DataFrame
            top5_data = df.head(30)
            top_5_data = top5_data.to_dict(orient='records')  # List of dicts for rendering

            # Get the number of rows and columns
            num_rows, num_cols = df.shape

            return render(request, 'upload.html', {
                'top_5_data': top_5_data,
                'file_name': uploaded_file.name,  # File name to display
                'num_rows': num_rows,
                'num_cols': num_cols
            })

        except Exception as e:
            # Handle errors (e.g., if the file is not a valid CSV)
            return render(request, 'upload.html', {'error_message': f"Error processing file: {str(e)}"})

    return render(request, 'upload.html')

def bert_analyze_sentiment(texts):
    # Tokenize the list of texts
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    
    # Make prediction with BERT model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Softmax to convert logits into probabilities
    probabilities = torch.softmax(logits, dim=-1)
    
    sentiments = []
    for prob in probabilities:
        sentiment_index = torch.argmax(prob).cpu().numpy()
        
        # Map the output to three sentiment categories
        if sentiment_index == 0 or sentiment_index == 1:
            sentiment = "negative"
        elif sentiment_index == 2:
            sentiment = "neutral"
        else:
            sentiment = "positive"

        print(sentiment)
        sentiments.append(sentiment)
    
    return sentiments[0]



def bert_model_analysis(request):
    try:
        # Retrieve the session data
        file_name = request.session.get('file_name', 'No file uploaded')
        num_rows = request.session.get('num_rows', 'Unknown')
        num_cols = request.session.get('num_cols', 'Unknown')

        # Retrieve the DataFrame from session
        df_json = request.session.get('df_data', None)

        if df_json:
            # Convert the JSON back to a DataFrame using StringIO
            data = StringIO(df_json)  # Wrap the JSON string in StringIO to simulate a file object
            df = pd.read_json(data, orient='split')
            print(df.head())  # Print the first few rows for debugging

            # Check if 'text' column exists
            if 'text' in df.columns:
                # Preprocess and analyze sentiment, store the results in a new column 'sentiment'
                df['bert_model_prediction'] = df['text'].apply(bert_analyze_sentiment) 
                
                updated_df_json = df.to_json(orient='split')
                request.session['df_data'] = updated_df_json # Assuming analyze_sentiment returns sentiment

                # Select only the relevant columns for display
                df_display = df[['text', 'cleaned_review', 'bert_model_prediction']]

                # For example, get the top 5 rows for display
                top_5_data = df_display.head(30).to_dict(orient='records')
            
                return render(request, 'bert_model.html', {
                    'file_name': file_name,
                    'num_rows': num_rows,
                    'num_cols': num_cols,
                    'top_5_data': top_5_data
                }) 
            else:
                return render(request, 'bert_model.html', {'error_message': "No 'text' column found in the dataset."})
        else:
            return render(request, 'bert_model.html', {
                'error_message': "No dataset in session"
            })

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message for debugging
        return render(request, 'bert_model.html', {'error': f"Error: {str(e)}"})
    
def nlp_analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    
    # Get the sentiment polarity score: -1 (negative) to +1 (positive)
    polarity = blob.sentiment.polarity

    # Map the polarity to the sentiment categories
    if polarity <= -0.1:
        return "negative"
    elif polarity >= 0.1:
        return "positive"
    else:
        return "neutral"
    
def nlp_model_analysis(request):
    try:
        # Retrieve the session data
        file_name = request.session.get('file_name', 'No file uploaded')
        num_rows = request.session.get('num_rows', 'Unknown')
        num_cols = request.session.get('num_cols', 'Unknown')

        # Retrieve the DataFrame from session
        df_json = request.session.get('df_data', None)

        if df_json:
            # Convert the JSON back to a DataFrame using StringIO
            data = StringIO(df_json)  # Wrap the JSON string in StringIO to simulate a file object
            df = pd.read_json(data, orient='split')
            print(df.head())  # Print the first few rows for debugging

            # Check if 'text' column exists
            if 'text' in df.columns:
                # Analyze sentiment using the simple NLP model (TextBlob)
                df['nlp_model_prediction'] = df['text'].apply(nlp_analyze_sentiment)  # Apply TextBlob-based sentiment analysis
                
                # Save the updated DataFrame back to the session
                updated_df_json = df.to_json(orient='split')
                request.session['df_data'] = updated_df_json  # Save the DataFrame with predictions

                # Select only the relevant columns for display
                df_display = df[['text', 'cleaned_review', 'nlp_model_prediction']]

                # For example, get the top 5 rows for display
                top_5_data = df_display.head(30).to_dict(orient='records')
            
                return render(request, 'nlp_model.html', {
                    'file_name': file_name,
                    'num_rows': num_rows,
                    'num_cols': num_cols,
                    'top_5_data': top_5_data
                })
            else:
                return render(request, 'nlp_model.html', {'error_message': "No 'text' column found in the dataset."})
        else:
            return render(request, 'nlp_model.html', {
                'error_message': "No dataset found in session"
            })

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error message for debugging
        return render(request, 'nlp_model.html', {'error': f"Error: {str(e)}"})

    

def show_sentiment_distribution(request):
    try:
        # Retrieve the DataFrame from session (assuming it's in JSON format in the session)
        df_json = request.session.get('df_data', None)

        if not df_json:
            return render(request, 'sentiment.html', {
                'error_message': "No dataset found in session."
            })

        # Convert the JSON back to a DataFrame using StringIO
        from io import StringIO
        data = StringIO(df_json)  # Wrap the JSON string in StringIO to simulate a file object
        df = pd.read_json(data, orient='split')

        # Create dictionaries to store sentiment counts for both models
        sentiment_counts_bert = {}
        sentiment_counts_nlp = {}

        # Check if 'bert_model_prediction' exists and get its value counts
        if 'bert_model_prediction' in df.columns:
            sentiment_counts_bert = df['bert_model_prediction'].value_counts()

        # Check if 'nlp_model_prediction' exists and get its value counts
        # if 'nlp_model_prediction' in df.columns:
        #     sentiment_counts_nlp = df['nlp_model_prediction'].value_counts()

        # If both sentiment_counts are empty, return an error message
        # if sentiment_counts_bert.empty and sentiment_counts_nlp.empty:
        if sentiment_counts_bert.empty:
            return render(request, 'sentiment.html', {
                'error_message': "No sentiment data found in the DataFrame."
            })

        # Create the BERT sentiment distribution graph
        fig_bert, ax_bert = plt.subplots(figsize=(5.5, 4))  # Individual figure for BERT
        bars_bert = ax_bert.bar(sentiment_counts_bert.index, sentiment_counts_bert.values, color=['green', 'gray', 'red'])
        ax_bert.set_title('BERT Sentiment Distribution')
        ax_bert.set_xlabel('Sentiment')
        ax_bert.set_ylabel('Count')
        ax_bert.tick_params(axis='x', rotation=0)

        # Add bar values (counts) on top of each bar for BERT
        ax_bert.bar_label(bars_bert, label_type='edge', padding=3)

        # Save the BERT plot to a BytesIO object for embedding in the HTML
        buf_bert = BytesIO()
        plt.savefig(buf_bert, format='png')
        buf_bert.seek(0)
        img_str_bert = base64.b64encode(buf_bert.read()).decode('utf-8')

        # Create the NLP sentiment distribution graph
        # fig_nlp, ax_nlp = plt.subplots(figsize=(5.5, 4))  # Individual figure for NLP
        # bars_nlp = ax_nlp.bar(sentiment_counts_nlp.index, sentiment_counts_nlp.values, color=['blue', 'orange', 'purple'])
        # ax_nlp.set_title('NLP Sentiment Distribution')
        # ax_nlp.set_xlabel('Sentiment')
        # ax_nlp.set_ylabel('Count')
        # ax_nlp.tick_params(axis='x', rotation=0)

        # Add bar values (counts) on top of each bar for NLP
        # ax_nlp.bar_label(bars_nlp, label_type='edge', padding=3)

        # Save the NLP plot to a BytesIO object for embedding in the HTML
        # buf_nlp = BytesIO()
        # plt.savefig(buf_nlp, format='png')
        # buf_nlp.seek(0)
        # img_str_nlp = base64.b64encode(buf_nlp.read()).decode('utf-8')

        # Prepare sentiment data for the template
        sentiment_data = {
            'bert': sentiment_counts_bert.to_dict(),
            # 'nlp': sentiment_counts_nlp.to_dict()
        }

        # Render the page with the sentiment data and both graphs
        return render(request, 'sentiment.html', {
            'sentiment_graph_bert': img_str_bert,  # Base64 encoded image for BERT
            # 'sentiment_graph_nlp': img_str_nlp,  # Base64 encoded image for NLP
            'sentiment_counts': sentiment_data,  # Sentiment counts for both BERT and NLP
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return render(request, 'sentiment.html', {'error_message': f"Error: {str(e)}"})
    

def prediction(request):
    if request.method == "POST":
        input_text = request.POST.get('text', '').strip()  # Get the text entered by the user

        if not input_text:
            return render(request, 'prediction.html', {'error': 'Please enter some text'})

        try:
            # Call your sentiment analysis functions for both models
            bert_sentiment = bert_analyze_sentiment(input_text)
            # nlp_sentiment = nlp_analyze_sentiment(input_text)

            # Pass both predictions to the template
            return render(request, 'prediction.html', {
                'input_text': input_text,
                'bert_prediction': bert_sentiment,
                # 'nlp_prediction': nlp_sentiment
            })
        except Exception as e:
            # If any error occurs, display the error message
            return render(request, 'prediction.html', {'error': str(e)})

    else:
        # Handle case where the form is not submitted via POST
        return render(request, 'prediction.html', {'error': 'Invalid request method'})