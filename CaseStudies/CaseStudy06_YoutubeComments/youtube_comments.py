"""Download YouTube comments to comments.csv (video_id, title, text, time).

Usage
-----
    python youtube_comments.py --video-id <VIDEO_ID> [--video-id <ID2> ...]
    python youtube_comments.py --keyword "machine learning" [--pages 3]

Prerequisites (see README.md for the full walk-through)
-------------------------------------------------------
1. A Google Cloud project with the YouTube Data API v3 enabled.
2. An OAuth "desktop" client secret JSON, downloaded from Google Cloud
   Console. Point the GOOGLE_CLIENT_SECRETS environment variable at it;
   never commit the file — .gitignore already covers client_secret*.json.
3. pip install google-api-python-client google-auth-oauthlib

Output schema (single, unified): video_id,title,text,time
comments_ml.py consumes exactly this file.
"""

import argparse
import csv
import os
import pickle

from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
CSV_FIELDS = ['video_id', 'title', 'text', 'time']
TOKEN_FILE = 'token.pickle'


def get_authenticated_service(client_secrets, insecure_dev=False):
    """OAuth2 flow with a cached token; first run opens a browser."""
    if insecure_dev:
        # localhost dev WITHOUT https; the *reason* this is opt-in is that
        # disabling transport security is fine for a classroom and wrong
        # anywhere else. Never enable in deployed code.
        os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

    credentials = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as f:
            credentials = pickle.load(f)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not client_secrets or not os.path.exists(client_secrets):
                raise SystemExit(
                    'Client secrets file not found. Set GOOGLE_CLIENT_SECRETS '
                    'to your downloaded OAuth JSON (see README.md).')
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
            # OOB "copy/paste the code" flow was removed by Google; the local
            # server flow is the supported replacement for desktop scripts
            credentials = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(credentials, f)

    return build('youtube', 'v3', credentials=credentials)


def get_video_comments(service, video_id, max_pages=3):
    """Top-level comments of one video, newest first, up to N pages."""
    comments = []
    page_token = None
    try:
        for page in range(max_pages):
            response = service.commentThreads().list(
                part='snippet', videoId=video_id, textFormat='plainText',
                maxResults=100, pageToken=page_token).execute()
            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({'text': snippet['textDisplay'],
                                 'time': snippet['publishedAt']})
            page_token = response.get('nextPageToken')
            if not page_token:
                break
    except HttpError as e:
        if b'commentsDisabled' in e.content:
            print(f'  comments disabled for {video_id}; skipping')
            return []
        if b'quotaExceeded' in e.content:
            raise SystemExit('YouTube API quota exhausted; retry tomorrow '
                             'or request more quota in Google Cloud Console.')
        raise
    return comments


def search_videos(service, keyword, max_pages=3, per_page=25):
    """Video ids/titles matching a keyword (search.list is quota-hungry)."""
    videos, page_token = [], None
    for _ in range(max_pages):
        response = service.search().list(
            q=keyword, part='id,snippet', type='video',
            maxResults=per_page, pageToken=page_token).execute()
        videos.extend({'video_id': item['id']['videoId'],
                       'title': item['snippet']['title']}
                      for item in response['items'])
        page_token = response.get('nextPageToken')
        if not page_token:
            break
    return videos


def write_csv(rows, outfile='comments.csv'):
    with open(outfile, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {outfile} ({len(rows)} comments)')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video-id', action='append', default=[],
                        help='YouTube video id; repeatable')
    parser.add_argument('--keyword', help='download comments from search hits')
    parser.add_argument('--pages', type=int, default=3,
                        help='max comment/search pages to fetch per video')
    parser.add_argument('--outfile', default='comments.csv')
    parser.add_argument('--insecure-dev', action='store_true',
                        help='allow OAuth over http for localhost development')
    args = parser.parse_args()

    if not args.video_id and not args.keyword:
        parser.error('give at least one --video-id or a --keyword')

    client_secrets = os.environ.get('GOOGLE_CLIENT_SECRETS', 'client_secret.json')
    service = get_authenticated_service(client_secrets,
                                        insecure_dev=args.insecure_dev)

    videos = list(args.video_id)
    titles = {}
    if args.keyword:
        hits = search_videos(service, args.keyword, max_pages=args.pages)
        videos.extend(v['video_id'] for v in hits)
        titles = {v['video_id']: v['title'] for v in hits}

    rows = []
    for video_id in videos:
        print(f'fetching comments for {video_id}...')
        for comment in get_video_comments(service, video_id, args.pages):
            rows.append({'video_id': video_id,
                         'title': titles.get(video_id, ''),
                         'text': comment['text'],
                         'time': comment['time']})

    write_csv(rows, args.outfile)


if __name__ == '__main__':
    main()
