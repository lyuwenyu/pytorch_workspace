import argparse

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--n', type=int, nargs='+', help='ss')
parser.add_argument('--s', type=str, default='ss')
args = parser.parse_args()


if __name__ == '__main__':
    print(args.n)
    print(args.s)