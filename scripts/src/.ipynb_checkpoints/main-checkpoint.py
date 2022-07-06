from conf import *

if __name__ == '__main__':
    if args.mode == 'train':
        from train import main
        main()
    elif args.mode == 'convert':
        from convert import main
        main()
    else:
        from predict import main
        main()