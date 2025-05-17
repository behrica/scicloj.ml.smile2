(ns scicloj.ml.smile2.scicloj.ml.smile2-test
  (:require
   [clojure.java.io :as io]
   [clojure.test :refer [deftest is]] 
   [scicloj.metamorph.ml :as ml] 
   [scicloj.metamorph.ml.toydata]
   [scicloj.ml.smile2.smile2]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod])
  (:import
   [smile.math.kernel HellingerKernel]
   [smile.classification MLP]
   [smile.math TimeFunction]
   [smile.base.mlp
    ActivationFunction
    Cost
    HiddenLayerBuilder
    LayerBuilder
    OutputFunction
    OutputFunction
    OutputLayerBuilder]
   [smile.base.rbf RBF]))




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


(defn- validate-test-train 
  
  ( [model-name params ds target-col]
   
   (let [m (ml/train ds
                     (assoc params
                            :model-type (keyword
                                         (str "smile2.classification/" model-name))))]

     (def m m)
     (def ds ds)
     (def target-col target-col)
     (is (= (ds/row-count ds)
            (count (get (ml/predict ds m) target-col))))))
  ([model-name params]
   (validate-test-train model-name params iris :species)
   )
  )

(def hidden-layer-builder
  (HiddenLayerBuilder. 1 1.0 (ActivationFunction/linear)))

(def output-layer-builder
  (OutputLayerBuilder. 3  OutputFunction/LINEAR  Cost/MEAN_SQUARED_ERROR))

(def rbfs
  (-> iris cf/feature ds/rowvecs
      (->> (map double-array))
      into-array
      (RBF/fit 10)))

(deftest train-predict-round-trip []
  ;(validate-test-train "svm" {:kernel (HellingerKernel.) :C 100.0})
  (validate-test-train "rbfnet" {:neurons rbfs})
  (validate-test-train "gbm" {})
  (validate-test-train "fld" {})
  (validate-test-train "cart" {})
  (validate-test-train "random-forest" {})
  ;(validate-test-train "mlp" {:builders [hidden-layer-builder output-layer-builder]})
  (validate-test-train "rda" {:alpha 0.5 :priori nil})
  (validate-test-train "knn" {})
  (validate-test-train "logit" {})
  (validate-test-train "lda" {})
  (validate-test-train "adaboost" {})
  (validate-test-train "qda" {}))


(deftest test-maxent


  (let [maxent-data
        (->
         (ds/->dataset {:x1 [1 2 3]
                        :x2 [4 5 6]
                        :y [1 2 3]})
         (ds-mod/set-inference-target :y))
        model
        (ml/train
         maxent-data
         {:p 7
          :lambda 0.1
          :model-type :smile2.classification/maxent})]
    (is (= [1 2 3]
           (:y
            (ml/predict maxent-data model))))))


(comment 

  (validate-test-train "mlp" {:builders
                              (into-array
                               LayerBuilder
                               [hidden-layer-builder output-layer-builder])})
  


  (def breast-cancer
    (->
     (ds/->dataset "resources/breastcancer.csv")
     (ds/categorical->number ["diagnosis"])
     (ds-mod/set-inference-target "diagnosis")))
  

  (def layers
    (into-array
     LayerBuilder
     [(smile.base.mlp.Layer/input 10)
      (smile.base.mlp.Layer/sigmoid 60)
      (smile.base.mlp.Layer/mle  1 OutputFunction/SIGMOID)]))
  

  (validate-test-train "mlp" {:builders
                              (into-array
                               LayerBuilder
                               [hidden-layer-builder output-layer-builder])})
  


  (ml/train breast-cancer


            {:model-type :smile2.classification/mlp
             :builders layers
             :epochs 10
             :eta (smile.math.TimeFunction/linear 0.2 1000 0.1)
             :alpha (TimeFunction/constant 0.2)
           ;:lambda
             })
  
  (def x
    (->
     (cf/feature breast-cancer)
     (ds/rowvecs)
     (#(into-array (map double-array %)))))
  

  (def y
    (int-array
     (->>
      (cf/target breast-cancer)
      (ds/rowvecs)
      (map first)
   ;
      )))
  
  


  (let [net (smile.classification.MLP. layers)]
    (.setLearningRate net (smile.math.TimeFunction/linear 0.2 1000 0.1))
    (.setMomentum net (TimeFunction/constant 0.2))
    (.setWeightDecay net 0.1)
    (dotimes [i 10] (.update net x y))
    )
  

  (smile.classification/mlp x y layers
                            10
                            (smile.math.TimeFunction/linear 0.2 1000 0.1)
                            (TimeFunction/constant 0.2)
                            0.1
                            )
  )


(deftest svm

  (let [ breast-cancer
        (->
         (ds/->dataset "resources/breastcancer.csv")
         (ds/categorical->number ["diagnosis"])
         (ds/update-column "diagnosis"
                           (fn [col]
                             (map
                              #(case %
                                 0 +1
                                 1 -1)
                              col)))
         
         (ds-mod/set-inference-target "diagnosis"))]
    (validate-test-train "svm" {:kernel (HellingerKernel.)
                                :C 100.0}
                         breast-cancer
                         "diagnosis")))



  
  
