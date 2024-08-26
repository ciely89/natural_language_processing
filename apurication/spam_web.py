from flask import Flask, render_template, request
import pickle
import MeCab
import numpy as np
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# ファイル名
data_file = "apurication\ok-spam.pickle"
model_file = "apurication\ok-spam-model.pickle"
label_names = ['OK', 'SPAM']

# 単語辞書を読み出す --- (※2)
data = pickle.load(open(data_file, "rb"))
word_dic = data[2]

# MeCabの準備
tagger = MeCab.Tagger()

# 学習済みモデルを読み出す --- (※3) 
model = pickle.load(open(model_file, "rb"))

# テキストがスパムかどうか判定する --- (※4)
def check_spam(text):
    # テキストを単語IDのリストに変換し単語の頻出頻度を調べる
    zw = np.zeros(word_dic['__id'])
    count = 0
    s = tagger.parse(text)
    # 単語毎の回数を加算 --- (※5)
    for line in s.split("\n"):
        if line == "EOS" or line == "":
            continue
        try:
            parts = line.split("\t")
            if len(parts) < 2:
                continue  # タブ区切りが足りない行をスキップ
            features = parts[1].split(",")
            if len(features) < 7:
                continue  # 原型が見つからない場合スキップ
            org = features[6]  # 単語の原型を取得
            if org in word_dic:
                id = word_dic[org]
                zw[id] += 1
                count += 1
        except IndexError as e:
            print(f"例外が発生しました: {e}, 行: {line}")
            continue
    if count > 0:
        zw = zw / count #  --- (※6)
        # 予測
        pre = model.predict([zw])[0] #  --- (※7)
        return label_names[pre]
    else:
        return "単語が見つかりませんでした"

# ルートページの処理
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        result = check_spam(input_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
