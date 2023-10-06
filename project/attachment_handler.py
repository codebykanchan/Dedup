import re
import requests
import io
import pandas as pd
from PIL import Image
from pytesseract import pytesseract

def find_links(string):
    '''
    Uses regular expression to find image URL
    '''
    regex = r"(http(s)?:\/\/[a-zA-Z0-9@:%._\+~#=\-\/]*(.[jJ][pP][eE][gG]|.[pP][nN][gG]|.[jJ][pP][gG]))"
    url = re.findall(regex,string)      
    return [x[0] for x in url]

def get_image_text(str): #text with attachment links as input
    '''
    Fetch text from attachment images
    '''
    img_lnks = find_links(str)
    all_text=''

    if len(img_lnks) > 0 :        
        for path in img_lnks:
            try:
                #print(f'Fetching image -> {path}')
                r = requests.get(path, allow_redirects=True)
                img = Image.open(io.BytesIO(r.content))
                text = pytesseract.image_to_string(img)
                all_text = all_text + ' ' + text
            except Exception as e:
                print (f'Exception -> {e}')
                print(f'Failed to fetch text from image {path}')
                
    return all_text