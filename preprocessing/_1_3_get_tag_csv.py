import pandas
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Select Tag Columns from a CSV')
    parser.add_argument('--input', '-p',  type=str,
                        default="../data/questions/Questions54FilteredNA.csv",
                        required=True, help='input csv file path')
    parser.add_argument('--output', '-o',  type=str,
                        default='../data/tags/20211110/Tags54.csv',
                        required=True, help='output csv file path')
    args = parser.parse_args()
    input_file = args.input
    output_file = args.output
    # df = pandas.read_csv('../data/csv/Question54.csv',lineterminator='\n')
    df = pandas.read_csv(input_file)
    header = ["Tags"]
    df.to_csv(output_file, columns=header)


if __name__ == "__main__":
    main()
