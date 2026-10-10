# Twitter Sentiment Analysis: COVID-19 Vaccination Tweets

Classifies tweets about COVID-19 vaccination as **positive**, **neutral** or **negative**, and compares linear
text classifiers on the task.

## Pipeline

1. **Load** `vaccination_tweets.csv`. The dataset is not included in this repo.
2. **Clean** each tweet: lowercase it, strip URLs, `#` symbols and punctuation, drop NLTK English stop
   words, then apply Porter stemming.
3. **Label** with TextBlob polarity: `< 0` is negative, `0` is neutral, `> 0` is positive. These labels are
   weak supervision, not human annotation.
4. **Explore** the sentiment distribution and a word cloud for each class.
5. **Featurise** with a `CountVectorizer` over unigrams and bigrams.
6. **Train and evaluate** logistic regression and a linear SVM, each tuned with `GridSearchCV`, using
   accuracy, a classification report and a confusion matrix.

## Results (held-out test set, 2,109 tweets)

| Model | Test accuracy |
|---|---|
| Logistic regression | 84.6% |
| Logistic regression + `GridSearchCV` | 85.9% |
| Linear SVM | 87.3% |
| Linear SVM + `GridSearchCV` | **87.6%** |

Negative is the hardest class: its recall is about 0.46 even for the best model, because it is the smallest
class.

## Files

| File | Contents |
|---|---|
| `Twitter sentiment analysis_live.ipynb` | Notebook with outputs and plots |
| `Twitter sentiment analysis_live.py` | The same pipeline as a script |

## Run

```bash
pip install pandas numpy nltk textblob wordcloud scikit-learn seaborn matplotlib
python -m nltk.downloader stopwords punkt
python "Twitter sentiment analysis_live.py"   # expects vaccination_tweets.csv in the working directory
```
