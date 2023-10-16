# ----------------------------------------------#
# Pro    : cbct
# File   : dataset.py
# Date   : 2023/2/22
# Author : Qing Wu
# Email  : wuqing@shanghaitech.edu.cn
# ----------------------------------------------#
import Polyner
import commentjson as json

if __name__ == '__main__':

    # load config
    # -----------------------
    with open("config.json") as config_file:
        config = json.load(config_file)

    # train
    # -----------------------
    for i in range(10):
        Polyner.train(img_id=i, config=config)