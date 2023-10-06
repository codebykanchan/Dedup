import json
import requests

JSON_FILE = 'angularjs.json'
COMMENT_FILE = 'angular_comments.txt'
DATA_FILE_PATH = './'
START_POS = 12286
MAX_ISSUES = 5000


with open(DATA_FILE_PATH + JSON_FILE, mode='r', encoding='utf-8') as f:
    jlst = json.load(f)
    num_items = len(jlst)
    with open(DATA_FILE_PATH + str(START_POS) +'_' + COMMENT_FILE, mode='w', encoding='utf-8') as f1:
        lastpos = min(START_POS+MAX_ISSUES, num_items)
        workset = jlst[START_POS : lastpos]
        for item in workset:
            if int(item['comments']) > 0:
                print(f"fetching comment -> {item['comments_url']}")
                req = requests.get(item['comments_url'], headers={"Authorization": "token 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"})
                f1.write(req.text + '\n')
