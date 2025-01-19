(ns scicloj.ml.smile2.scicloj.ml.smile2-test
  (:require [clojure.test :refer [deftest is]]
            [scicloj.ml.smile.protocols]
            [scicloj.metamorph.ml.toydata]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.libs.smile.data]
            [scicloj.ml.smile2.scicloj.ml.smile2])
  (:import [smile.math.kernel HellingerKernel]
           [smile.base.mlp Layer OutputFunction HiddenLayerBuilder ActivationFunction Cost LayerBuilder OutputFunction OutputLayerBuilder]))



(def iris
  (scicloj.metamorph.ml.toydata/iris-ds))

(deftest gbm
  (let [m (ml/train iris {:model-type :smile2.classification/gbm})]
    (is (= {0 50, 1 50, 2 50}
           (frequencies (:species (ml/predict iris m)))))))


(deftest adaboost
  (let [m (ml/train iris {:model-type :smile2.classification/adaboost})]
    (is (= {0 50, 1 50, 2 50}
           (frequencies (:species (ml/predict iris m)))))))

(defn- validate-test-train [model-name params]
  (let [m (ml/train iris
                    (assoc params
                           :model-type (keyword
                                        (str "smile2.classification/" model-name))))]
    (is (= 150
           (count (:species (ml/predict iris m)))))))

(def hidden-layer-builder
  (HiddenLayerBuilder. 1 0.0 (ActivationFunction/linear)))

(def output-layer-builder
  (OutputLayerBuilder. 3  OutputFunction/LINEAR  Cost/MEAN_SQUARED_ERROR))


(deftest train-predict-round-trip []
  ;(validate-test-train "svm" {:kernel (HellingerKernel.) :C 100.0})
  ;(validate-test-train "rbfnet" {})
  (validate-test-train "gbm" {})
  (validate-test-train "fld" {})
  (validate-test-train "cart" {})
  (validate-test-train "random-forest" {})
  ;(validate-test-train "mlp" {:builders [hidden-layer-builder output-layer-builder]})
  (validate-test-train "rda" {:alpha 0.5 :priori nil})
  (validate-test-train "knn" {})
  (validate-test-train "logit" {})
  (validate-test-train "lda" {})
  ;(validate-test-train "maxent" {:p 4})
  (validate-test-train "adaboost" {})
  (validate-test-train "qda" {}))

