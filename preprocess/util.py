import csv
import re
from datetime import date
from datetime import datetime


def load_tags(tags_fpath):
    import pandas as pd
    tags = []
    df = pd.read_csv(tags_fpath)
    for idx, row in df.iterrows():
        tags.append(row['tag'])
    print("# tags = %s" % len(tags))
    return tags


def load_tag_cnt(tag_cnt_dict_fpath):
    print("loading tag cnt dict...")
    tag_cnt_dict = {}
    with open(tag_cnt_dict_fpath) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            tag = row[0]
            cnt = row[1]
            tag_cnt_dict[tag] = int(cnt)
    return tag_cnt_dict


def write_dict_to_csv(dict_tmp, csv_fpath, header=None):
    with open(csv_fpath, 'w') as myfile:
        wr = csv.writer(myfile)
        if header:
            wr.writerow(header)
        for k in dict_tmp:
            try:
                wr.writerow([k, dict_tmp[k]])
            except Exception as e:
                print("Error %s" % e)
    print("Write %s successfully!" % csv_fpath)


def write_list_to_csv(list_tmp, csv_fpath, header):
    if type(list_tmp) == set:
        list_tmp = list(list_tmp)
    with open(csv_fpath, 'w') as myfile:
        wr = csv.writer(myfile)
        if header:
            wr.writerow(header)
        for x in list_tmp:
            try:
                if type(x) == str:
                    x = [x]
                wr.writerow(x)
            except Exception as e:
                print("Error %s" % e)
    print("Write %s successfully!" % csv_fpath)


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
    """[summary]

    Args:
        raw_html ([type]): [description]

    Returns:
        [type]: [description]
    """
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


def get_date():
    from datetime import date
    today = date.today().strftime("%b-%d-%Y")
    return today


def get_datetime():
    """[summary]
    """
    from datetime import datetime
    dt_string = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
    print("date and time =", dt_string)


def save_pickle(data, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def vocab_to_index_dict(vocab, ifpad=False):
    # use sort because require consistent
    vocab_list = list(vocab)
    vocab_list.sort()
    vocab_dict = dict()
    for i in range(len(vocab_list)):
        vocab_dict[vocab_list[i]] = i
    if ifpad:
        vocab_dict['<PAD>'] = len(vocab_dict)
    return vocab_dict


def indexing_label(label):
    vocab = label_vocab
    embedd_zero = [0 for i in range(len(vocab)+1)]
    for l in label:
        if l in vocab:
            embedd_zero[vocab[l]] = 1
    return np.array(embedd_zero)