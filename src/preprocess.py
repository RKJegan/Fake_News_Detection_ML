import nltk


# Download necessary NLTK resources(only once is enough)
"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"""

import re
import string 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # Convert to lowercase

    text = re.sub(r"http\S+|www\S+|https\S+", "",text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"\d+", "", text)  # Remove digits
    text = text.translate(str.maketrans("","",string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization and remove stopwords

    return " ".join(words)


if __name__ == "__main__":
    sample_text = "This is a sample text for preprocessing! Visit https://example.com for more info."
    processed_text = preprocess_text(sample_text)
    print("Original Text:", sample_text)
    print("Processed Text:", processed_text)