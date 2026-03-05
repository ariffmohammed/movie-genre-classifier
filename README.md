# Movie Genre Classifier

A machine learning project that predicts the genre of a movie based on its description. Built entirely independently using patterns learned from previous projects.

## How It Works

Movie descriptions are converted to numerical features using TF-IDF vectorization. Logistic Regression then learns the patterns between word importance scores and genre. Given a new description it predicts whether the movie is action, comedy, horror, or romance.

## What I Learned

- Multiclass classification with more than 2 output labels
- Applying a known ML pipeline independently to a new problem
- How the same text classification approach works across different domains

## Tech Stack

- Python
- scikit-learn

## Dataset

20 hardcoded movie descriptions across 4 genres. Full version will use a real world movie dataset with thousands of labeled descriptions.
