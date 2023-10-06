
def clean_string(lstr):
    '''
    This function cleans the string, removes special characters.
    '''
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