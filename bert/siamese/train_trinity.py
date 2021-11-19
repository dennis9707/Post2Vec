from train import get_train_args, init_train_env
import logging
import os
import sys
from sklearn import preprocessing


sys.path.append("../")
sys.path.append("../../")


logger = logging.getLogger(__name__)


def main():
    args = get_train_args()

    # Build Tag Vocab
    seed_everything(args.seed)
    tab_vocab_path = args.vocab_file
    tag_vocab = pd.read_csv(tab_vocab_path)
    tag_list = tag_vocab["tag"].astype(str).tolist()
    mlb = preprocessing.MultiLabelBinarizer()
    mlb.fit([tag_list])
    args.mlb = mlb
    input_train = args.train_data_file
    input_valid = args.valid_data_file
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    model = init_train_env(args, tbert_type='twin')
    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    train_examples = load_examples(
        train_dir, model=model, num_limit=args.train_num)
    valid_examples = load_examples(
        valid_dir, model=model, num_limit=args.valid_num)
    train(args, train_examples, valid_examples, model, train_with_neg_sampling)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
