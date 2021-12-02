import sys
sys.path.append('../../')
from data_structure.question import Question, NewQuestion
import pandas as pd
from util.util import get_files_paths_from_directory

def main():

    target_dir = "../data/processed_train"
    files = get_files_paths_from_directory(target_dir)

    print(files)
    for file in files:
        dataset = pd.read_pickle(file)
        print(1)
        print(len(dataset))
        for question in dataset:
            title = question.get_title()
            text = question.get_text()
            code = question.get_code()

if __name__ == '__main__':
    main()