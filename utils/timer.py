import time
import inspect


def timer():
    start = time.time()

    def end(method_name="Unnamed function"):

        print(method_name + " took : " + str(time.time() - start) + " seconds.")
        return

    return end
