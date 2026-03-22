import re
from urllib.parse import urlparse


def extract_features(url):
    try:
        parsed = urlparse(url)

        features = []

        # 1. URL length
        features.append(len(url))

        # 2. Has HTTPS
        features.append(1 if parsed.scheme == "https" else 0)

        # 3. Domain length
        features.append(len(parsed.netloc))

        # 4. Number of dots
        features.append(url.count("."))

        # 5. Number of hyphens
        features.append(url.count("-"))

        # 6. Number of digits
        features.append(len(re.findall(r"\d", url)))

        # 7. Has IP address
        features.append(1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0)

        # 8. Has suspicious words
        suspicious_words = ["login", "verify", "secure", "update", "bank", "account"]
        features.append(1 if any(word in url.lower() for word in suspicious_words) else 0)

        # 9. URL path length
        features.append(len(parsed.path))

        # 10. Number of query params
        features.append(url.count("?"))

        return features

    except Exception as e:
        print("Feature extraction error:", str(e))
        return [0]*10   # ALWAYS return fixed size