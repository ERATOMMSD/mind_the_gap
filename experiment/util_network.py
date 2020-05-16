from typing import *
import requests
import json
import datetime
import random
import socket

try:
    with open("slack_webhook.txt", "r") as f:
        slack_url = f.readline().strip()
except Exception as e:
    pass



def notify_slack(message: str) -> bool:
    try:
        requests.post(slack_url, data=json.dumps({
            'text': message,  # 投稿するテキスト
            'username': u'rnn2wfa',  # 投稿のユーザー名
            'icon_emoji': u':ghost:',  # 投稿のプロフィール画像に入れる絵文字
            'link_names': 1,  # メンションを有効にする
        }))
        return True
    except Exception as e:
        return False


def get_time_hash():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "-" \
           + socket.gethostbyname(socket.gethostname()).replace(".", "-")


