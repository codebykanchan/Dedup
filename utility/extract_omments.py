import json


COMMENT_FILE = 'comments.txt'
COMMENT_DATA_FILE = 'comments.csv'
DATA_FILE_PATH = './'


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
    
    
with open(DATA_FILE_PATH + COMMENT_FILE, mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    with open(DATA_FILE_PATH + COMMENT_DATA_FILE, mode='w', encoding='utf-8') as f1:
        f1.write('number,comment\n')
        for line in lines:
            try:
                jlst = json.loads(line)
                issue_url = jlst[0]['issue_url']
                index = issue_url.rfind('/') + 1
                issue_id = issue_url[index:]
                comment = ''
                for item in jlst:
                    comment += item['body'] + '#;;'
                    
                comment = cleanString(comment)
                f1.write(issue_id + ',' + comment + '\n')
            except:
                print(f'failed to process {line}')

