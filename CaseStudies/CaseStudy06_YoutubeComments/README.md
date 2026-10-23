# Case Study 06 — YouTube comments → simple NLP

Two halves:

1. **Get the data** (`youtube_comments.py`): authenticate to the YouTube
   Data API v3 and download comments to a single tidy CSV
   (`video_id,title,text,time`).
2. **Learn from it** (`comments_ml.py`): TF-IDF features + logistic
   regression over labelled comments — a compact introduction to text
   classification, runnable offline on the committed sample.

## Part 1 — the scraper

### One-time Google Cloud setup

1. Create a project at <https://console.cloud.google.com>.
2. Enable the **YouTube Data API v3** for it.
3. Create **OAuth credentials** of type *Desktop app*, and download the JSON
   as `client_secret....json`.
4. Point the environment variable at it (and do NOT commit the file —
   `.gitignore` already blocks `client_secret*.json`):

   ```bash
   export GOOGLE_CLIENT_SECRETS=/path/to/client_secret_....json
   ```

### Dependencies and usage

```bash
pip install google-api-python-client google-auth-oauthlib
python youtube_comments.py --video-id dQw4w9WgXcQ
python youtube_comments.py --keyword "machine learning" --pages 3
```

The first run opens a browser for OAuth consent (this uses the
`run_local_server` flow — Google's copy/paste "out-of-band" flow was
deprecated), then caches `token.pickle` locally (also git-ignored).
Output: `comments.csv` in the unified schema.

### Quotas and etiquette

- The default daily quota (10,000 units) is consumed fast: commentThreads
  reads cost 1 unit, search.list costs **100 units per call**. Start with
  `--video-id` before hammering `--keyword`.
- Respect YouTube's [Terms of Service](https://www.youtube.com/static?template=terms)
  and [`robots.txt`](https://www.youtube.com/robots.txt): use the API (not
  raw scraping), don't republish personal data, and filter public figures
  respectfully when doing NLP on comments.

## Part 2 — the ML exercise

```bash
python comments_ml.py --input comments_sample.csv   # offline exercise
python comments_ml.py --input comments.csv          # after labelling real data
```

- `comments_sample.csv` is committed: 96 synthetic comments with hand-written
  **synthetic labels** (positive/negative tone) so the workflow runs without
  any scraping or labelling effort. Treat them as a teaching placeholder —
  real labels are your job.
- On real data you create `label` yourself: hand-label a few hundred rows
  (positive/negative/toxic/spam/promotional...), then retrain.
- Exercise ideas: change `ngram_range`, try character n-grams
  (`analyzer='char_wb'`) for misspellings, swap in `LinearSVC`, or evaluate
  with cross-validation instead of the single split. Watch out for the
  usual leakage: the vectorizer must be fit on the training text only (the
  script already does this).

Expected output on the sample set: ≈ 0.9 accuracy with indicative-words
report — small, synthetic, but *real end-to-end ML over text*.
