import sys
import feature_engineering.standardizer.standardizer as dt


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]
    # you want your project to do.
    dt.run(args)


if __name__ == "__main__":
    main()
