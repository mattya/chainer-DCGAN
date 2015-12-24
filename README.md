## You can try web demo [here](http://mattya.github.io/chainer-DCGAN/) !!

# chainer-DCGAN
Chainer implementation of Deep Convolutional Generative Adversarial Network (http://arxiv.org/abs/1511.06434)

## 説明
画像を生成するニューラルネットです。<br>
12/24のchainer advent calendarに解説を書きました。 http://qiita.com/mattya/items/e5bfe5e04b9d2f0bbd47 <br>
このコードは現在試行錯誤の途中であり、突然の変更などの可能性が十分あります。ご了承ください。

## 使い方(暫定)
* chainer 1.5が必要
* 学習済みモデルから生成のみを行うには、visualizer.pyを使用する。GPU無くてもOK。
``` python visualizer.py ```
* 学習を行うにはDCGAN.pyを実行する。image_dir変数で指定されたディレクトリに、学習元となる画像ファイルを置く。GPUが必要で、何時間かかかる。

## サンプル
20万枚の顔イラスト画像で約3時間学習を行った結果(GTX 970使用)。
<img src="https://raw.githubusercontent.com/mattya/chainer-DCGAN/master/sample4.png" height="800px">

特定の画像の生成元となったベクトルzにノイズを加えると、髪型や服装などが少しずつ異なる画像を生成できる。
このことから、本モデルが過学習しているわけではない(特定の画像を暗記しているわけではない)ことが示唆される。
<img src="https://raw.githubusercontent.com/mattya/chainer-DCGAN/master/sample2.png" height="600px">

画像間の連続的変換。
<img src="https://raw.githubusercontent.com/mattya/chainer-DCGAN/master/sample3.png" height="600px">

## 参考文献
本家の実装です。モデルの相違点はleaky_reluの代わりにeluを使っているくらいです。 https://github.com/soumith/dcgan.torch




