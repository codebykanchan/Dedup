import requests
import json

BASE_URL = "https://api.github.com/repos/angular/angular.js/issues"
ISSUE_STATE = 'all'
PER_PAGE_COUNT = 100
START_PAGE = 1
ISSUE_COUNT = 17100
DATA_FILE = 'angularjs.txt'
DATA_JSON = 'angularjs.json'
DATA_CSV = 'angularjs.csv'
DATA_FILE_PATH = './'
DUPLICATE_KEY = 'resolution: duplicate'

def toJson(filepath, lst):
    with open(filepath, mode='w',encoding='utf-8') as f1:
        f1.write('[\n')
        maxitems = len(lst)
        itemcount = 0
        for item in lst:
            lstr = json.dumps(item)
            f1.write(lstr)
            itemcount += 1
            if itemcount < maxitems:
                f1.write(',\n')
        f1.write('\n]')

def cleanString(lstr):
    lstr = lstr.replace('\r','')
    lstr = lstr.replace('\n','')
    lstr = lstr.replace(',','')
    lstr = lstr.replace('`','')
    lstr = lstr.replace('\'','')
    lstr = lstr.replace('"','')
    lstr = lstr.replace('{','')
    lstr = lstr.replace('}','')
    lstr = lstr.replace('%','')
    lstr = lstr.replace('&','')
    return lstr
    
def getComments(link):
    print(f'fetching comments from {link}')
    req = requests.get(link, headers={"Authorization": "token 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"})
    jreq = json.loads(req.text)
    comment = ''
    for com in jreq:
        comment += com['body'] + ';'
        
    comment = comment.replace('\r','')
    comment = comment.replace('\n','')
    comment = comment.replace(',','')
    return comment

def extractDataToCsv(lst):
    with open(DATA_FILE_PATH + DATA_CSV, mode='w',encoding='utf-8') as f1:
        f1.write('Id,Number,Title,LabelsNames,LabelDescriptions,State,CreatedDate,ClosedDate,IsDraft,IssueType,Description,StateReason,AttachmentText,Duplicate,Comments,SimilarityScore')
        for item in lst:
            is_dup = False
            itemstr = str(item['id'])+','
            itemstr += str(item['number']) + ','
            
            title = ('' if item['title'] is None else item['title'] )
            title = cleanString(title)
            itemstr += title + ','
            
            item_lable_names = ''
            item_label_descp = ''
            for lbl in item['labels']:
                item_lable_names += lbl['name'] + ';'
                if lbl['name'] == DUPLICATE_KEY :
                    is_dup = True
                item_label_descp += ('' if lbl['description'] is None else (lbl['description'] + ';' ))
                
            item_label_descp = cleanString(item_label_descp)
            itemstr += item_lable_names + ',' + item_label_descp + ','
            
            itemstr += item['state'] + ','
            itemstr += item['created_at'] + ','
            itemstr += ('' if item['closed_at'] is None else item['closed_at'] ) + ','
            itemstr += ('Y' if 'draft' in item.keys() and item['draft'] else 'N') + ','
            itemstr += ('PR' if 'pull_request' in item.keys() else '' ) + ','
            
            body = ('' if item['body'] is None else item['body'])
            body = cleanString(body)
            itemstr += body + ','
            
            itemstr += ('' if item['state_reason'] is None else item['state_reason']) + ',,'
            itemstr += ('Y' if is_dup else '') + ',,'
            
            f1.write('\n'+itemstr)
    
#with open(DATA_FILE_PATH + DATA_FILE, mode='w', encoding='utf-8') as f:
#    for i in range(START_PAGE,ISSUE_COUNT//PER_PAGE_COUNT + 1):
#        URL_QUERY = f'?state={ISSUE_STATE}&page={i}&per_page={PER_PAGE_COUNT}'
#        print("Scraping; -> " + BASE_URL + URL_QUERY)
#        req = requests.get(BASE_URL + URL_QUERY, headers={"Authorization": "token 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"})
#        f.write(req.text + '\n')

jlist = []
with open(DATA_FILE_PATH + DATA_FILE, mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        jlst = json.loads(line)
        jlist += jlst

toJson(DATA_FILE_PATH + DATA_JSON, jlist)
extractDataToCsv(jlist)
