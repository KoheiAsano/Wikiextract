# Wikiextract_Embedding

日本語テキスト群のユニークな単語をベンチマークと一緒にベクトル化するpy.作成したモデルを読み込み二次元まで落として可視化するjupyter


- embedding_uniquewords.py  
[wikiextractor](https://github.com/attardi/wikiextractor)を利用して取得したwikipedia記事全体をプールとして、日本語テキスト群(デフォルトはtexts/以下CSV形式)を対象にTF-IDFでそのテキスト群のユニークな固有名詞(mecab-ipadic-neologdで分類されるもの)を抽出し、ベンチマークとする辞書登録語と一緒に100次元ベクトルにembedする  
出力はdata/以下、

- Vis_Embedding.ipynb  
embedding_uniquewords.pyで生成したベクトルを二次元に落としmatplotlibでplot  
![example](example.png)
- normalize.py  
上記プログラムで利用する正規化プログラム(全角、小文字スペル、数字を0へ置換)  

- wiki_AA_text  
wikiextractorで抜き出した日本語wikitext群  

- texts  
Embeddingしたいtextを入れるディレクトリ

# **環境構築**
---

MeCabのインストールと(使う場合)mecab-ipadic-neologdのインストールが必要です。
[MeCab](http://taku910.github.io/mecab/)  
[neologd](https://github.com/neologd/mecab-ipadic-neologd)  

必要ライブラリのインストール(足りないのがあれば必要に応じてpip installしてください。)
```
$ pip install  -r requirements.txt
```  

加えて、デフォルトではテキストを形態素解析する際に、デフォルトはUni系MeCab-ipadic-neologdのPathで設定しているため、Windows環境では注意


# **使用方法**
---

- embedding_uniquewords.py  

```
$ python embedding_uniquewords.py texts
```  


- Vis_Embedding.ipynb  
jupyterセルをそれぞれ実行  


# **参考**
---

[gensim embedding](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/)  
[wikiextractor](https://github.com/attardi/wikiextractor)  
