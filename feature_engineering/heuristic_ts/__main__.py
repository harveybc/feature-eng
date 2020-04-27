import sys
import feature_engineering.heuristic_ts.heuristic_ts as dt


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]
    # you want your project to do.
    dt.run(args)


if __name__ == "__main__":
    main()
