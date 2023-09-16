import pandas as pd
src_path = "../new_data/csv/Posts.csv"
for i,chunk in enumerate(pd.read_csv(src_path, chunksize=500000)):
    chunk.to_csv('../new_data/split_csv/posts_chunk{}.csv'.format(i), index=False)
    
    
    
# file = pd.read_csv('../new_data/split_csv/posts_chunk1.csv')
# file = file.drop(columns=['OwnerUserId', 'LastEditorUserId',
#                           'ViewCount','AnswerCount','CommentCount','ContentLicense','Score','OwnerDisplayName',"LastEditorDisplayName",
#                           "CommunityOwnedDate","ClosedDate","LastEditDate","LastActivityDate","FavoriteCount"])

# # handle question
# from util import clean_html_tags, clean_html_tags2, write_dict_to_csv
# questions = []

# for index, row in file.iterrows():
#     if row['PostTypeId'] == 1:
#         question = {"Id": row["Id"],
#                     "CreationDate": row["CreationDate"],
#                     "Title": ' '.join(clean_html_tags(row["Title"]).split()),
#                     "Body": ' '.join(clean_html_tags(row["Body"]).split()),
#                     "Tags": row["Tags"],
#                     }
#         print(question)
#         question.append(question)
#         break
    