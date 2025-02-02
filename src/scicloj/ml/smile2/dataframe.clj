(ns scicloj.ml.smile2.dataframe
  (:require
   [clojure.java.io :as io]
   [scicloj.metamorph.ml.toydata]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dt])
  
  (:import
   [smile.data DataFrame]
   [smile.data.vector ValueVector]
   [smile.datasets CPU]
   )

  
  )


(defn value-vector->column [vv-column]
  (let [vv-type (.. vv-column dtype id name)
        col-name (.name vv-column)
        stream
        (case vv-type
          "Float" (.doubleStream vv-column)
          "Double" (.doubleStream vv-column)
          "Int" (.intStream vv-column))]
    (ds/new-column col-name (stream-seq! stream))))

(defn df->ds [df]
  (ds/new-dataset
   (map
    value-vector->column
    (.columns df))))







(defn ds->df [ds]
  (def ds ds)
  (DataFrame.
   (into-array ValueVector
               (map
                (fn [col]
                  (def col col)
                  (let [col-name (str (-> col meta :name))
                        convert-fn
                        (case
                         (-> col meta :datatype)
                          :int16  dt/->int-array
                          :float64 dt/->double-array
                          )
                        
                        ]
                    (def convert-fn convert-fn)
                    (def col-name col-name)
                    (ValueVector/of col-name  
                                    (convert-fn col))))

                (ds/columns ds)))))


(comment
  (def iris
    (scicloj.metamorph.ml.toydata/iris-ds))
  


  (def cpu
    (.data (CPU.
            (.toPath
             (io/file "cpu.arff")))))
  


  (-> iris
      ds->df
      df->ds)
  

  (df->ds cpu)
  )


(comment
  (import '[smile.data.type StructField]
          '[smile.data.type DoubleType]
          '[smile.data.type DataType])
  
  (def col
    (->
     (scicloj.metamorph.ml.toydata/iris-ds)
     :sepal_width))
  
  (def vv
    (reify ValueVector
      (field [this]
        (StructField.
         (-> col meta :name str)
         (DataType/of DoubleType)))
      (size [this]
        (dt/ecount col))
      (getInt [this i]
        (int
         (dt/get-value col i)))
      (getDouble [this i]
        (dt/cast
         (dt/get-value col i)))
      (get [this ^int i]
        (dt/get-value col i))))
  
  (def df
    (DataFrame. (into-array [vv])))

  
  )