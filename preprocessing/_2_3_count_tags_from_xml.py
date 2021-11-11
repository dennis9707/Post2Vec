import xml.etree.ElementTree as Xet
import pandas as pd


def main():
    rows = []
    xmlparse = Xet.parse("../data/tags/Tags.xml")
    root = xmlparse.getroot()
    for i in root:
        id = i.attrib["Id"]
        tag_name = i.attrib["TagName"]
        count = i.attrib["Count"]
        row = {"id": id,
               "tag_name": tag_name,
               "count": int(count),
               }

        rows.append(row)
    cols = ["id", "tag_name", "count"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(by='count', ascending=False)
    df2 = df.reset_index(drop=True)
    df2.to_csv("../data/tags/Tag.csv")


if __name__ == "__main__":
    main()
