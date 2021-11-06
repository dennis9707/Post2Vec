import xml.etree.ElementTree as Xet
import pandas as pd
import time
import logging
from tqdm import tqdm


def separate_text_code(html_str):
    import re
    # regex: <pre(.*)><code>([\s\S]*?)</code></pre>
    regex_pattern = r'<pre(.*?)><code>([\s\S]*?)</code></pre>'
    code_list = []
    html_text = html_str
    for m in re.finditer(regex_pattern, html_str):
        # print("start %d end %d" % (m.start(), m.end()))
        raw_code = html_str[m.start():m.end()]
        clean_code = clean_html_tags(raw_code).replace('\n', ' ')
        code_list.append(clean_code)
        # remove code
        html_text = html_text.replace(raw_code, " ")
    clean_html_text = clean_html_tags(html_text)
    clean_html_text = remove_symbols(clean_html_text)
    if len(code_list) == 0:
        code_str = ''
    else:
        code_str = ' '.join(code_list)
    return clean_html_text, code_str


def clean_html_tags(raw_html):
    from bs4 import BeautifulSoup
    try:
        text = BeautifulSoup(raw_html, "html.parser").text
    except Exception as e:
        # UnboundLocalError
        text = clean_html_tags2(raw_html)
    finally:
        return text


def clean_html_tags2(raw_html):
    import re
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def remove_symbols(strtmp):
    return strtmp.replace('\n', ' ')

def check_nullable(tag, parent):
    if tag in parent.attrib:
        return parent.attrib[tag]
    else:
        return ""
# Parsing the XML file
rows =[]
def xml2csv(i):
    if i.attrib['PostTypeId']=='1':
        Id = i.attrib["Id"]
        AcceptedAnswerId = check_nullable("AcceptedAnswerId", i)
        CreationDate = i.attrib["CreationDate"]
        Score = i.attrib["Score"]
        ViewCount = i.attrib["ViewCount"]
        Body,Code = separate_text_code(i.attrib["Body"])
        LastActivityDate = i.attrib["LastActivityDate"]
        FavoriteCount = check_nullable("FavoriteCount", i)
        AnswerCount = check_nullable("AnswerCount", i)
        CommentCount = check_nullable("CommentCount", i)
        Title = i.attrib["Title"]
        Tags = clean_html_tags(i.attrib["Tags"])
        if Tags == "":
            Tags = i.attrib["Tags"]
        row = {"Id": Id,
                "AcceptedAnswerId": AcceptedAnswerId,
                "CreationDate": CreationDate,
                "Score": Score,
                "ViewCount": ViewCount,
                "Body": Body,
                "Code": Code,
                "LastActivityDate": LastActivityDate,
                "FavoriteCount": FavoriteCount,
                "AnswerCount": AnswerCount,
                "CommentCount": CommentCount,
                "Title": Title,
                "Tags": Tags,
                }
        rows.append(row)
  


def main():
    input_file = "../data/post_data/Posts1.xml"
    logging.info("Start to process files...")
    xmlparse = Xet.parse(input_file)
    root = xmlparse.getroot()
    length = len(root.findall('row'))

    for i in tqdm(root):
        xml2csv(i)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()