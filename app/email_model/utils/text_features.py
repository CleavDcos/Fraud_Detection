#to convert text to numerical values for models
#this text will be coming from email


import re
from sklearn.feature_extraction.text import TfidfVectorizer

#typoes of keywords
URGENT_WORDS = ["urgent", "immediately", "asap", "now"]
THREAT_WORDS = ["suspended", "blocked", "terminated", "restricted"]
SUSPICIOUS_PHRASES = ["click here", "verify now", "login now", "update account"]



def count_keywords(text, keywords):
    return sum(1 for word in keywords if word in text)


def extract_keyword_features(text):
    return {
        "urgent_count": count_keywords(text, URGENT_WORDS),
        "threat_count": count_keywords(text, THREAT_WORDS),
        "suspicious_phrase_count": count_keywords(text, SUSPICIOUS_PHRASES),
    }


#this checks for features of text wguich includes characters
def extract_structural_features(text):
    words = text.split()

    uppercase_words = sum(1 for w in words if w.isupper())
    special_chars = len(re.findall(r"[!@#$%^&*(),.?\":{}|<>]", text))

    return {
        "email_length": len(text),
        "uppercase_words": uppercase_words,
        "special_char_count": special_chars,
    }


#text to vectors, tf-idf is machine learn patterns
def get_tfidf_vectorizer():
    return TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english"
    )