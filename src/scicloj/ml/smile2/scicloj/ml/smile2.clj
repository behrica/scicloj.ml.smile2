(ns scicloj.ml.smile2.scicloj.ml.smile2
  (:require
   [scicloj.ml.smile.protocols]
   [scicloj.metamorph.ml.toydata]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds]
   [tech.v3.libs.smile.data]
   [smile.classification :as classification])
  (:import (smile.data.formula Formula)
           [smile.base.cart SplitRule]))






(defn- train-with-formula [feature-ds target-ds train-fn model-params]
  (let [ds (ds/append-columns feature-ds target-ds)
        df (tech.v3.libs.smile.data/dataset->smile-dataframe ds)
        formula
        (Formula/of
         (-> target-ds ds/column-names first name)
         (into-array  (->> feature-ds ds/column-names (map name))))]
    
    {:train-type :formula
     :model 
     (apply train-fn
            formula
            df
            model-params)}))

(defn- train-with-x-y [feature-ds target-ds train-fn model-params]
  {:train-type :x-y
   :model (apply
           train-fn
           (into-array (map double-array (-> feature-ds ds/rowvecs)))
           (int-array (-> target-ds vals first))
           model-params)})



(defn predict-from-formula [model-data ds]
  (let [df (tech.v3.libs.smile.data/dataset->smile-dataframe ds)]
    (.predict (:model model-data) df)))

(defn- predit-from-x-y [model ds]
  
  (.predict
   model
   (into-array (map double-array
                    (-> ds

                        ds/rowvecs)))))

  
  

(defn define-smile-model! [var-definition default-options]
  
  (let [model-type (keyword "smile2.classification" (name (:symbol var-definition)))]
    (ml/define-model! model-type
      (fn [feature-ds target-ds options]
        (let [supported-options
              (map keyword
                   (drop 2 (:args-with-options var-definition)))
              train-fn (-> var-definition :var)
              model-params
              (map
               #(get options %1 %2)
               supported-options
               default-options)]

          

          (case (vec (take 2 (:args-with-options var-definition)))
            [formula data] (train-with-formula feature-ds target-ds train-fn model-params)
            [x y] (train-with-x-y feature-ds target-ds train-fn model-params)))
        
        )
      (fn [feature-ds thawed-model {:keys [options model-data target-categorical-maps target-columns] :as model}]
        (let [ds (ds/add-column
                  
                  feature-ds
                  (ds/new-column 
                   (first target-columns)
                   (repeat (ds/row-count feature-ds) nil))
                  )
              
              
              
              prediction
              (case (:train-type model-data)
                :formula (predict-from-formula model-data ds )
                :x-y (predit-from-x-y (:model model-data) feature-ds) 
                )
              ]
          
          (ds/new-dataset
           [( ds/new-column (keyword (first target-columns))
              prediction
              {:column-type :prediction})
            ]))
        
        )
      {})))


(def var-definitions
  (map (fn [[s v]]
         (hash-map :symbol s
                   :var v
                   :args-with-options (-> v meta :arglists second)))
       (ns-publics (find-ns 'smile.classification))))

(def default-options
  {'gbm   [500 20 6 5 0.05 0.7]
   'adaboost [500 20 6 1]
   'rbfnet [nil false] ;neurons
   'fld [-1 0.0001]
   'svm [nil nil 1E-3] ;kernel C
   'cart [SplitRule/GINI 20 2 5]
   'random-forest [500 0 SplitRule/GINI 20 500 5 1.0]
   'mlp [nil 10 0.1 0.0 0.0] ; builders
   'rda [nil nil 0.0001] ;alpha
   'knn [1]
   'logit [0.0 1E-5 500]
   'lda [nil 0.0001]
   'maxent [nil 0.1 1E-5 500] ;p
   'qda [nil 0.0001]})


(run!
 #(define-smile-model! % (get default-options (:symbol %)))
 var-definitions
 )


