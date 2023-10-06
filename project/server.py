from fastapi import Request, FastAPI
import json_parser
import attachment_handler
import comment_handler
import json
import pandas as pd
from io import StringIO
import text_pre_processor
import project_model
from pydantic import BaseModel
from typing import Any, Dict, AnyStr, List, Union
from fastapi.encoders import jsonable_encoder


'''
Line 1: We import FastAPI, which is a Python class that provides all the functionality for the API.

Line 26: We create an instance of the class FastAPI and name it app. This is the app referred to by <uvicorn> in the above command.

Line 66: We create a POST path.

Line 67: We define the function "get_body" that will execute whenever someone visits the above path.

Line 92: We return a response to the client whenever the route is accessed.
'''
app = FastAPI()

class Item(BaseModel):
    '''
    Json payload structure definition
    '''
    url: str
    repository_url: str
    labels_url: str
    comments_url: str
    events_url: str
    html_url: str
    id: int
    node_id: str
    number: int
    title: Union[str, None] = None
    user: Union[dict[AnyStr, Any], None] = None
    labels: list[Any]
    state: Union[str, None] = None
    locked: bool = False
    assignee: Union[dict[AnyStr, Any], None] = None
    assignees: list[Any]
    milestone: Union[dict[AnyStr, Any], None] = None
    comments: int
    created_at: str
    updated_at: Union[str, None] = None
    closed_at: Union[str, None] = None
    author_association: Union[str, None] = None
    active_lock_reason: Union[Any, None] = None
#    draft: bool = False
#    pull_request: dict[AnyStr, Any]
    body: Union[str, None] = None
    closed_by: Union[dict[AnyStr, Any], None] = None
    reactions: Union[dict[AnyStr, Any], None] = None
    timeline_url: Union[str, None] = None
    performed_via_github_app: Union[Any, None] = None
    state_reason: Union[Any, None] = None



@app.post("/items/")
async def get_body(request: Item):
#    print(type(request))
#    print(request)

    jitem = jsonable_encoder(request)

#    jitem = json.loads(req)
    data = 'Id,Number,Title,LabelsNames,LabelDescriptions,State,CreatedDate,ClosedDate,IsDraft,IssueType,Description,StateReason,AttachmentText,Duplicate,Comments,SimilarityScore\n'
    data += json_parser.extract_data_from_json(jitem)

    data = StringIO(data)

    issue_df = pd.read_csv(data, sep=',')
    issue_df = issue_df.fillna('')
    
    issue_df.loc[0,['AttachmentText']]  = attachment_handler.get_image_text(issue_df['Description'][0])
    issue_df.loc[0,['Comments']] = comment_handler.get_comments(jitem)

    columns = ['Title','Description','AttachmentText','Comments']

    text_pre_processor.clean_pipeline(issue_df, columns)

    vec = project_model.get_model_vector(issue_df)
    similar =  project_model.get_similar(vec)
    project_model.save_data(jitem['number'], vec, similar)
    return similar
