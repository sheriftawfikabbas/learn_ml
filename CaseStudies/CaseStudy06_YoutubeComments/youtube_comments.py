import pandas as pd
import csv
import os
import pickle
import google.oauth2.credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
import sys

CLIENT_SECRETS_FILE = "client_secret_57986039921-03joetttcf9cobqpmmn40570ehu4027e.apps.googleusercontent.com.json"

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


def get_authenticated_service():
    credentials = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    #  Check if the credentials are invalid or do not exist
    if not credentials or not credentials.valid:
        # Check if the credentials have expired
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_console()

        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)

    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)


def get_video_comments(service, **kwargs):
    comments = []
    results = service.commentThreads().list(**kwargs).execute()
    i = 0
    while results:
        for item in results['items']:
            comment = {"text": item['snippet']['topLevelComment']['snippet']['textDisplay'],
                       "time": item['snippet']['topLevelComment']['snippet']['publishedAt']}
            comments.append(comment)
            print("Comment ", i)
            i += 1

        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments


def write_to_csv(comments):
    with open('comments.csv', 'w') as comments_file:
        comments_writer = csv.writer(
            comments_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        comments_writer.writerow(['Video ID', 'Title', 'Comment'])
        for row in comments:
            # convert the tuple to a list and write to the output file
            comments_writer.writerow(list(row))


def get_videos(service, **kwargs):
    final_results = []
    results = service.search().list(**kwargs).execute()

    i = 0
    max_pages = 3
    while results and i < max_pages:
        final_results.extend(results['items'])

        # Check if another page exists
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
            results = service.search().list(**kwargs).execute()
            i += 1
        else:
            break

    return final_results


def search_videos_by_keyword(service, **kwargs):
    results = get_videos(service, **kwargs)
    final_result = []
    for item in results:
        title = item['snippet']['title']
        video_id = item['id']['videoId']
        comments = get_video_comments(
            service, part='snippet', videoId=video_id, textFormat='plainText')
        # make a tuple consisting of the video id, title, comment and add the result to
        # the final list
        final_result.extend([(video_id, title, comment)
                             for comment in comments])

    write_to_csv(final_result)


def get_video_by_id(service, video_id):
    comments = get_video_comments(
        service, part='snippet', videoId=video_id, textFormat='plainText')

    c = open('comments.csv', 'w')
    for comment in comments:
        c.write(comment+'\n')
        c.flush()
    c.close()


# When running locally, disable OAuthlib's HTTPs verification. When
# running in production *do not* leave this option enabled.
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
service = get_authenticated_service()

if len(sys.argv) > 1:
    videoId = sys.argv[1]

    comments = get_video_comments(
        service, part='snippet', videoId=videoId, textFormat='plainText')

    # search_videos_by_keyword(service, q=keyword, part='id,snippet', eventType='completed', type='video')

    comments_df = pd.DataFrame(comments)
    comments_df.to_csv('comments.csv')