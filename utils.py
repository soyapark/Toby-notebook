import pickle
import random

def _preprocess(user_data):
    if "index" in user_data:
        del user_data["index"]
    if "initial" in user_data:
        del user_data["initial"]
    if "testset" in user_data:
        del user_data["testset"]

    for c in range(len(user_data["content"])):
        user_data["content"][c] = user_data["content"][c].split("mnist_png")[1]

    return user_data

def _convert_type(user_data):
    for i in range(len(user_data["actual"])):
        user_data["result"][i] = int(user_data["result"][i])
        if user_data["actual"][i] != 'unknown':
            user_data["actual"][i] = int(user_data["actual"][i])

def _print_out_code(_input, user_data, i=True):
    _compare = _input
    if i:
        _compare = _compare + "_train40_pandas.pickle"
    else:
        _compare = _compare + "_eval"

    with open(_compare, "rb") as f:
        data = pickle.load(f)
        
        cnt = 0
        for i in range(len(data["actual"])):
            if str(data["actual"][i]) == str(user_data["result"][i]):
                # print(user_data["result"][i])
                cnt = cnt +1

        print( str(random.randint(1,10100000)) + "135" + str(int(cnt / len(data["actual"]) * 100)) )

from IPython.display import Image
from google.colab.patches import cv2_imshow
import cv2

def show_image(img_name):
    img_name = img_name.split('mnist_png')[1]
    testim=cv2.imread("mnist_png"+img_name)
    return cv2_imshow(testim)