import json_parser
import attachment_handler
import comment_handler
import json
import requests
import pandas as pd
from io import StringIO
import text_pre_processor
import project_model

BASE_URL = "https://api.github.com/repos/angular/angular.js/issues/17045"

req = requests.get(BASE_URL, headers={"Authorization": "token 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"}).text
jitem = json.loads(req)

data = 'Id,Number,Title,LabelsNames,LabelDescriptions,State,CreatedDate,ClosedDate,IsDraft,IssueType,Description,StateReason,AttachmentText,Duplicate,Comments,SimilarityScore\n'
data += json_parser.extract_data_from_json(jitem)

data = StringIO(data)

issue_df = pd.read_csv(data, sep=',')

issue_df.loc[0,['AttachmentText']]  = attachment_handler.get_image_text(issue_df['Description'][0])
issue_df.loc[0,['Comments']] = comment_handler.get_comments(jitem)

columns = ['Title','Description','AttachmentText','Comments']

text_pre_processor.clean_pipeline(issue_df, columns)

vec = project_model.get_model_vector(issue_df)
print(project_model.get_similar(vec));