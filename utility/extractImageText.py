import re
import requests
import io
import pandas as pd
from PIL import Image
from pytesseract import pytesseract

INPUT_CSV = 'angularjs+comments.csv'
INPUT_DIR = './data/'
OUTPUT_CSV = 'angularjs_imagetext.csv'

def findLinks(string):
    regex = r"(http(s)?:\/\/[a-zA-Z0-9@:%._\+~#=\-\/]*(.[jJ][pP][eE][gG]|.[pP][nN][gG]|.[jJ][pP][gG]))"
    url = re.findall(regex,string)      
    return [x[0] for x in url]

df = pd.read_csv(INPUT_DIR + INPUT_CSV)

for indx in df.index:
    descstr = str(df['Description'][indx])
    imglnks = findLinks(descstr)
    print(f'Working on index -> {indx}')
    if len(imglnks) > 0 :        
        alltext=''
        for path in imglnks:
            try:
                print(f'Fetching image -> {path}')
                r = requests.get(path, allow_redirects=True)
                img = Image.open(io.BytesIO(r.content))
                text = pytesseract.image_to_string(img)
                alltext = alltext + ' ' + text
            except Exception as e:
                print(f'Failed to fetch text from image {path}')
        df['AttachmentText'][indx]= alltext
        
df.to_csv(INPUT_DIR + OUTPUT_CSV, index=False, encoding='utf-8')

