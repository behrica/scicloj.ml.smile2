# scicloj.ml.smile2/scicloj.ml.smile2

Plugin of Java [Smile](https://haifengl.github.io/) (> v.2.6) into [metamorph.ml](https://github.com/scicloj/metamorph.ml)



Usage:
```
(require '[scicloj.metamorph.ml :as ml])

(def iris (scicloj.metamorph.ml.toydata/iris-ds))

;; train model on some data
(def model (ml/train iris {:model-type :smile2.classification/gbm}))

;; predict on some (usualy other) data
(ml/predict iris model)
```

See here for the integation of Smile version 2.6 into metamorph.ml: 
[scicloj.ml.smile](https://github.com/scicloj/scicloj.ml.smile)

Distributed under the GPL 3.0
