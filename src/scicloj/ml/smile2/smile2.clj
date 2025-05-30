(ns scicloj.ml.smile2.smile2
  (:require 
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset :as ds] 
   [scicloj.ml.smile2.dataframe]
   [smile.classification]

   [tech.v3.dataset.column-filters :as cf])
  (:import
   [smile.base.cart SplitRule]
   (smile.data.formula Formula)))


(defn- train-with-formula [feature-ds target-ds train-fn model-params]
  (let [ds (ds/append-columns feature-ds target-ds)
        df (scicloj.ml.smile2.dataframe/ds->df ds)
        
        formula
        (Formula/of
         (-> target-ds ds/column-names first str)
         (into-array  (->> feature-ds ds/column-names (map str))))]
    
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

(defn- predict-from-formula [model-data ds]
  (let [df (scicloj.ml.smile2.dataframe/ds->df   
            (cf/feature ds))]
    (.predict (:model model-data) df)))


(defn- predit-from-x-y [model ds]
  
  (.predict
   model
   (into-array (map double-array
                    (-> ds

                        ds/rowvecs)))))

(defn- predict-maxent [model-data feature-ds]
  (.predict
                         (:model model-data)
                         (into-array (map int-array
                                          (-> feature-ds
                                              ds/rowvecs)))))

(defn- train-maxent [options feature-ds target-ds model-params]
  {:train-type :maxent
               :model 
               (apply smile.classification/maxent 
                      (:p options)
                      (into-array (map int-array (-> feature-ds ds/rowvecs)))
                      (int-array (-> target-ds vals first))
                      model-params
                      )})

  
  

(defn- define-smile-model! [var-definition default-options]
   
  (let [model-type (keyword "smile2.classification" (name (:symbol var-definition)))]
    (ml/define-model! model-type
      (fn [feature-ds target-ds options]
        (let [supported-options
              (map keyword
                   (drop 
                    (case model-type
                      :smile2.classification/maxent 3
                      2) 
                    
                    (:args-with-options var-definition)))
              train-fn (-> var-definition :var)
              model-params
              (map
               #(get options %1 %2)
               supported-options
               default-options)]

          

          (comment 
            (def default-options default-options)
            (def supported-options supported-options)
            (def options options)
            (def var-definition var-definition)
            (def model-params model-params)
            (def feature-ds feature-ds)
            (def target-ds target-ds)
            (def model-type model-type)
            (def train-fn train-fn)
            )
          
          (cond 
            (= (:var var-definition) #'smile.classification/maxent)
            (let [reduced-model-params (dissoc options :p)]
              ;(def reduced-model-params reduced-model-params)
              (train-maxent options feature-ds target-ds model-params))
            
            :else
            (let [first-wto-arguments (vec (take 2 (:args-with-options var-definition)))]
              (case first-wto-arguments
                [formula data] (train-with-formula feature-ds target-ds train-fn model-params)
                [x y] (train-with-x-y feature-ds target-ds train-fn model-params)))
            ))
        
        )
      (fn [feature-ds thawed-model {:keys [options model-data target-categorical-maps target-columns] :as model}]
        (let [ds-1 (ds/add-column

                    feature-ds
                    (ds/new-column
                     (first target-columns)
                     (repeat (ds/row-count feature-ds) nil)))

              prediction
              (case (:train-type model-data)
                :maxent (predict-maxent model-data feature-ds)
                :formula (predict-from-formula model-data feature-ds)
                :x-y (predit-from-x-y (:model model-data) feature-ds))]
          
          (ds/new-dataset
           [( ds/new-column  (first target-columns)
              prediction
              {:column-type :prediction})
            ]))
        
        )
      {})))


(def var-definitions
  (->> 
   (ns-publics (find-ns 'smile.classification))
  ;;  (remove (fn [[k v]] 
  ;;            (= 'mlp k)))
   (map (fn [[s v]]
          (hash-map :symbol s
                    :var v
                    :args-with-options (-> v meta :arglists second)))
        )))

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
   'maxent [ 0.1 1E-5 500] ;p
   'qda [nil 0.0001]})


(run!
 #(define-smile-model! % (get default-options (:symbol %)))
 var-definitions
 )














