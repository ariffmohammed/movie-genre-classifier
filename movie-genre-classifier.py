from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

reviews = [
    ("action", "An elite soldier battles enemy forces in a high stakes war zone filled with explosions and combat"),
    ("action", "A spy goes on a dangerous mission to stop a terrorist attack and save the world"),
    ("action", "A superhero fights a powerful villain threatening to destroy the entire city"),
    ("action", "A former marine goes on a revenge mission after his family is kidnapped by criminals"),
    ("action", "Two cops chase down a dangerous gang across the city in high speed car chases"),
    ("comedy", "A clumsy office worker accidentally becomes the boss and hilariously tries to hide it"),
    ("comedy", "Two best friends go on a road trip that turns into a series of ridiculous disasters"),
    ("comedy", "A man wakes up and relives the same embarrassing day over and over again"),
    ("comedy", "A group of friends throw a surprise party that completely spirals out of control"),
    ("comedy", "A shy accountant accidentally joins a dance competition and becomes a viral sensation"),
    ("horror", "A family moves into a haunted house and starts experiencing terrifying supernatural events"),
    ("horror", "A group of teenagers are hunted one by one by a masked killer in the woods"),
    ("horror", "A scientist experiments on himself and slowly transforms into a dangerous monster"),
    ("horror", "A young girl discovers she can see dead people and is terrorized by dark spirits"),
    ("horror", "Passengers on a cruise ship are trapped with a creature that hunts them at night"),
    ("romance", "Two strangers meet on a train and fall in love before reaching their destination"),
    ("romance", "A woman returns to her hometown and reconnects with her childhood sweetheart"),
    ("romance", "Two colleagues who pretend to date for a work event slowly fall for each other"),
    ("romance", "A man writes letters to his future wife not knowing she is already in his life"),
    ("romance", "Two people from rival families fall in love despite their families disapproval"),
]

labels = [label for label, msg in reviews]
texts = [msg for label, msg in reviews]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = [label for label, msg in reviews]

model = LogisticRegression()
model.fit(X, y)

def predict(msg):
    x = vectorizer.transform([msg])
    result = model.predict(x)[0]
    return f"Genre: {result}"

while True:
    user_input = input("Enter a movie description: ")
    print(predict(user_input))