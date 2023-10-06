import json
import requests
import helper

def download_comment(item, auth_token = ""): 
    '''
    This function downloads the comments from bugs/tickets.
    Parameters:
    item : Json Object
    auth_token : github authentication token
    '''
    req = None
    if int(item['comments']) > 0:
        if len(auth_token) > 0 :
            req = requests.get(item['comments_url'], headers={"Authorization": "token " + auth_token})
        else:
            req = requests.get(item['comments_url']).text

    return req

def get_comments(item, auth_token = ""):
    '''
    This function fetches the comments and converts to json format
    Parameters:
    item : Json Object
    auth_token : github authentication token
    '''
    comments = download_comment(item, auth_token)
    comment = ''

    if comments is not None:
        jlst = json.loads(comments)

        try:
            for item in jlst:
                comment += item['body'] + ' ; '

            comment = helper.clean_string(comment)
        except:
            None
                #print(f'failed to process {line}')
    return comment
