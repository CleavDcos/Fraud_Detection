import re
import math
from urllib.parse import urlparse

def has_ip(url):
    match = re.search(r'\d+\.\d+\.\d+\.\d+', url)
    return 1 if match else 0

def url_length(url):
    return len(url)

def count_dots(url):
    return url.count('.')

def has_at_symbol(url):
    return 1 if '@' in url else 0

def has_https(url):
    return 1 if url.startswith("https") else 0

def count_digits(url):
    return sum(c.isdigit() for c in url)

def has_suspicious_words(url):
    keywords = ['login', 'verify', 'bank', 'secure', 'account', 'update']
    return 1 if any(word in url.lower() for word in keywords) else 0

def subdomain_count(url):
    return url.count('.') - 1

def get_domain(url):
    try:
        return urlparse(url).netloc
    except:
        return ""
    
def domain_length(url):
    return len(get_domain(url))

def suspicious_tld(url):
    suspicious = ['.xyz', '.tk', '.ml', '.ga', '.cf']
    return 1 if any(tld in url for tld in suspicious) else 0

def has_hyphen(url):
    return 1 if '-' in get_domain(url) else 0

#detect randomness
def url_entropy(url):
    prob = [float(url.count(c)) / len(url) for c in dict.fromkeys(url)]
    entropy = -sum([p * math.log2(p) for p in prob])
    return entropy

def has_brand_name(url):
    brands = ['paypal', 'google', 'amazon', 'facebook', 'bank']
    return 1 if any(brand in url.lower() for brand in brands) else 0

def suspicious_word_count(url):
    keywords = ['login', 'verify', 'secure', 'account', 'update']
    return sum(word in url.lower() for word in keywords)

def is_long_domain(url):
    return 1 if len(get_domain(url)) > 25 else 0



def extract_features(url):
    return [
        url_length(url),
        has_ip(url),
        count_dots(url),
        has_at_symbol(url),
        has_https(url),
        count_digits(url),
        has_suspicious_words(url),
        suspicious_word_count(url),
        subdomain_count(url),
        url_entropy(url),
        domain_length(url),
        is_long_domain(url),
        suspicious_tld(url),
        has_hyphen(url),
        has_brand_name(url)
    ]
#Extracting each feature from the URL in order to perform non-linear classfication using the Random Forest Classifier.
#Random forest uses trees to make decisions based on each feature of the URL.