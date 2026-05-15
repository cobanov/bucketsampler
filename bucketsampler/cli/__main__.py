"""Allow ``python -m bucketsampler.cli`` to invoke the same entry as the
``bucketsampler`` console script.
"""

from bucketsampler.cli import main

if __name__ == "__main__":
    main()
