"""Introductory text classification over YouTube comments (TF-IDF + logistic regression).

Turns raw comments into numbers with TF-IDF and trains a binary sentiment
classifier. Works offline against the committed comments_sample.csv so the
exercise can be run before (or without) scraping real data.

    python comments_ml.py [--input comments.csv]

CSV schema: video_id,title,text,time[,label]. The label column is optional;
without it nothing can be trained, so a friendly explanation is printed
instead. The sample file's labels are SYNTHETIC teaching labels (is this a
nice or a nasty comment?), hand-written for the exercise.
"""

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='comments.csv',
                        help='comments CSV (default: comments.csv)')
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input)
    except OSError:
        raise SystemExit(f'cannot read {args.input}; scrape with '
                         'youtube_comments.py or use comments_sample.csv')

    print(f'{len(df)} comments from {args.input}')
    if 'label' not in df.columns:
        raise SystemExit('no `label` column — nothing to learn. '
                         'Label your data (e.g. positive/negative) first. '
                         'comments_sample.csv shows the idea.')

    df = df.dropna(subset=['text', 'label'])
    # keep it honest: never evaluate on data you trained on
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=0,
        stratify=df['label'])

    # TF-IDF: tfidf(t, d) = tf(t, d) * log(N / df(t)) — frequent words in a
    # document, down-weighted where they are common across ALL documents
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                 min_df=1)
    X_train_v = vectorizer.fit_transform(X_train)   # fit on TRAIN only
    X_test_v = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_v, y_train)
    y_pred = clf.predict(X_test_v)

    print(f'\naccuracy: {accuracy_score(y_test, y_pred):.3f} '
          f'(test size {len(y_test)})')
    print(classification_report(y_test, y_pred))

    # which words push the score towards each class?
    names = vectorizer.get_feature_names_out()
    weights = clf.coef_[0]
    top_pos = names[weights.argsort()[-10:][::-1]]
    top_neg = names[weights.argsort()[:10]]
    classes = list(clf.classes_)
    print(f'most indicative of "{classes[1]}": {list(top_pos)}')
    print(f'most indicative of "{classes[0]}": {list(top_neg)}')


if __name__ == '__main__':
    main()
