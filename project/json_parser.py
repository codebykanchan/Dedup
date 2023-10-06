import json
import helper

def extract_data_from_json(item):
    '''
    parse Json and extract the data
    parametres:
    item : json object 
    '''
    is_dup = False
    item_str = str(item['id'])+','
    item_str += str(item['number']) + ','
    
    title = ('' if item['title'] is None else item['title'] )
    title = helper.clean_string(title)
    item_str += title + ','
    
    item_lable_names = ''
    item_label_descp = ''
    for lbl in item['labels']:
        item_lable_names += lbl['name'] + ';'
        item_label_descp += ('' if lbl['description'] is None else (lbl['description'] + ';' ))
        
    item_label_descp = helper.clean_string(item_label_descp)
    item_str += item_lable_names + ',' + item_label_descp + ','
    
    item_str += item['state'] + ','
    item_str += item['created_at'] + ','
    item_str += ('' if item['closed_at'] is None else item['closed_at'] ) + ','
    item_str += ('Y' if 'draft' in item.keys() and item['draft'] else 'N') + ','
    item_str += ('PR' if 'pull_request' in item.keys() else '' ) + ','
    
    body = ('' if item['body'] is None else item['body'])
    body = helper.clean_string(body)
    item_str += body + ','
    
    item_str += ('' if item['state_reason'] is None else item['state_reason']) + ',,'
    item_str += ('Y' if is_dup else '') + ',,'
    
    return item_str