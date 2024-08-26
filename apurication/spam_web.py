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

# 学習用データの読み込み
data = pickle.load(open(data_file, "rb"))
word_dic = data[2]

tagger = MeCab.Tagger()

# 学習済みモデルを読み込む
model = pickle.load(open(model_file, "rb"))

# テキストがスパムかどうか判定する
def check_spam(text):

    zw = np.zeros(word_dic['__id'])
    count = 0
    s = tagger.parse(text)
    
    for line in s.split("\n"):
        if line == "EOS" or line == "":
            continue
        try:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            features = parts[1].split(",")
            if len(features) < 7:
                continue
            org = features[6]
            if org in word_dic:
                id = word_dic[org]
                zw[id] += 1
                count += 1
        except IndexError as e:
            print(f"例外が発生しました: {e}, 行: {line}")
            continue
    if count > 0:
        zw = zw / count
        # 予測
        pre = model.predict([zw])[0]
        return label_names[pre]
    else:
        return "単語が見つかりませんでした"

#ページ処理
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        result = check_spam(input_text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)