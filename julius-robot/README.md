# おかえり彼女 (okaeri-kanojo)

## 概要

「ただいま」の音声入力に対し「おかえり」と返答してくれるプログラムです。
話者認識機能で望まれない暴発を防ぎます。

## 環境

* Windows 10
* MinGW32 4.2.0 + MSYS 1.0
* Julius 4.4.1
* Chainer 1.16.0
    * ほか cuDNN, CUDA, Anaconda など必要に応じて

## ファイル構成

* Makefile
    * okaeri_kanojo.c コンパイル用
    * `JDIR` をご自身の環境に変更しておいてください
    * `make` でコンパイルできます
* classify.py
    * 話者認識判別スクリプト
    * `--testfile=[評価csv]` オプションで実行できます
* dataset.py
    * データセットを Chainer が利用できる形に変換するクラス
* mlp.py
    * mnist 記載の MLP クラスを外出したもの
* okaeri_kanojo.c
    * プログラム本体
    * コンパイルすると okaeri_kanojo.jpi になり Julius のプラグインとなる
* play_okaeri.py
    * 「おかえり」音声を再生するだけのスクリプト
    * スクリプト内、音声ファイルのパスを適宜書き換えてください
    * wav 再生ではなく音声合成にしたい場合も、本スクリプトを修正することで実現できます
* plugin_defs.h
    * Julius のプラグイン化に必要なヘッダファイル
* train.csv
    * 学習データ
    * 作者のデータ100サンプル、他人のデータ約100サンプル
* train.py
    * 学習用スクリプト

## 使う

以下準備が必要です。

* Julius 最新版のコンパイル + インストール
    * ディクテーションキットも
* Chainer のインストール
* 「おかえり」の音声ファイル

実行については、okaeri ディレクトリにて
```
[Julius のバイナリ] -C [ディクテーションキットのディレクトリ]/main.jconf -C [ディク(ry]/am-gmm.jconf -nolog -quiet -plugindir .
```

と打ち込んでください。あとはマイクに話せば動くはず。

for more information: http://qiita.com/tsubasaogawa/items/3830e4889cccbbb0cffa
