"""
Email: autuanliu@163.com
Date: 2018/9/26
"""

import argparse


def get_setting(args):
    if args.info is None:
        print('args is None.')
    else:
        print(f'args is {args.info}')


def main():
    parser = argparse.ArgumentParser(description='argparse example.')
    parser.add_argument('--info', type=str, default='tf', metavar='I', help='info message (default: \'tf\')')
    args = parser.parse_args()
    get_setting(args)


if __name__ == '__main__':
    main()
