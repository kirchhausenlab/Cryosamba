import sys
from time import sleep

for i in range(10):
    print(f"hello {i}", end="\r")
    sys.stdout.flush()
    sleep(1)  # Sleep to simulate some processing time
