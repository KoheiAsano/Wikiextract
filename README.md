# Wikiextract_Embedding

���{��e�L�X�g�Q�̃��j�[�N�ȒP����x���`�}�[�N�ƈꏏ�Ƀx�N�g��������py.�쐬�������f����ǂݍ��ݓ񎟌��܂ŗ��Ƃ��ĉ�������jupyter


- embedding_uniquewords.py  
[wikiextractor](https://github.com/attardi/wikiextractor)�𗘗p���Ď擾����wikipedia�L���S�̂��v�[���Ƃ��āA���{��e�L�X�g�Q(�f�t�H���g��texts/�ȉ�CSV�`��)��Ώۂ�TF-IDF�ł��̃e�L�X�g�Q�̃��j�[�N�ȌŗL����(mecab-ipadic-neologd�ŕ��ނ�������)�𒊏o���A�x���`�}�[�N�Ƃ��鎫���o�^��ƈꏏ��100�����x�N�g����embed����  
�o�͂�data/�ȉ��A

- Vis_Embedding.ipynb  
embedding_uniquewords.py�Ő��������x�N�g����񎟌��ɗ��Ƃ�matplotlib��plot  
![example](example.png)
- normalize.py  
��L�v���O�����ŗ��p���鐳�K���v���O����(�S�p�A�������X�y���A������0�֒u��)  

- wiki_AA_text  
wikiextractor�Ŕ����o�������{��wikitext�Q  

- texts  
Embedding������text������f�B���N�g��

# **���\�z**
---

MeCab�̃C���X�g�[����(�g���ꍇ)mecab-ipadic-neologd�̃C���X�g�[�����K�v�ł��B
[MeCab](http://taku910.github.io/mecab/)  
[neologd](https://github.com/neologd/mecab-ipadic-neologd)  

�K�v���C�u�����̃C���X�g�[��(����Ȃ��̂�����ΕK�v�ɉ�����pip install���Ă��������B)
```
$ pip install  -r requirements.txt
```  

�����āA�f�t�H���g�ł̓e�L�X�g���`�ԑf��͂���ۂɁA�f�t�H���g��Uni�nMeCab-ipadic-neologd��Path�Őݒ肵�Ă��邽�߁AWindows���ł͒���


# **�g�p���@**
---

- embedding_uniquewords.py  

```
$ python embedding_uniquewords.py texts
```  


- Vis_Embedding.ipynb  
jupyter�Z�������ꂼ����s  


# **�Q�l**
---

[gensim embedding](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/)  
[wikiextractor](https://github.com/attardi/wikiextractor)  
