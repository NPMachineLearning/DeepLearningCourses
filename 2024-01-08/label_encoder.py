from collections import Counter

from sklearn.preprocessing import LabelEncoder

planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Earth"]
encoder = LabelEncoder()
encoder.fit(planets)
encoded = encoder.transform(planets)
print(encoded)
print(Counter(planets))