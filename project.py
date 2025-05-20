# Install required libraries
!pip install -q transformers datasets nltk
# Import libraries
import pandas as pd
import nltk
import re
from transformers import pipeline
nltk.download('punkt')
nltk.download('stopwords')
# Sample mock social media data (can be replaced with live data later)
data = {
    'username': ['user1', 'user2', 'user3'],
    'text': [
    "I love this product! It's absolutely amazing 😍",
    "I'm so frustrated with the service. Never again!",
    "Feeling kinda meh about today... not bad, not great."
  ]
}
# Create a DataFrame
df = pd.DataFrame(data)
# Function to clean text
def clean_text(text):
  text = re.sub(r'http\S+', '', text) # remove URLs
  text = re.sub(r'@\w+', '', text) # remove mentions
  text = re.sub(r'#\w+', '', text) # remove hashtags
  text = re.sub(r'[^\w\s]', '', text) # remove punctuation
  text = text.lower().strip()
  return text
df['cleaned_text'] = df['text'].apply(clean_text)
# Load a sentiment analysis pipeline
sentiment_analyzer = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
# Apply sentiment analysis
df['emotion_result'] = df['cleaned_text'].apply(lambda x: sentiment_analyzer(x)[0])
# Extract emotion label and score
df['emotion'] = df['emotion_result'].apply(lambda x: x['label'])
df['confidence'] = df['emotion_result'].apply(lambda x: x['score'])
# Display results
df = df.drop(columns='emotion_result')
df