(ns scicloj.ml.smile2.dataframe
  (:require
   [tech.v3.dataset :as ds])

  (:import
   [smile.data DataFrame]
   [smile.data.vector ValueVector IntVector NullableIntVector NullableBooleanVector
    ObjectVector NullableFloatVector NullableLongVector NullableCharVector NullableByteVector
    StringVector]))


(defn- value-vector->column [vv-column]
  (let [vv-type
        (.. vv-column dtype id name)
        col-name (.name vv-column)
        the-seq
        (case vv-type
          "Float" (stream-seq! (.doubleStream vv-column))
          "Double" (stream-seq! (.doubleStream vv-column))
          "Int" (.intStream vv-column)
          "Boolean" (map #(case %
                            0 false
                            1 true
                            nil)
                         (stream-seq! (.intStream vv-column)))
          "Object" (stream-seq! (.stream vv-column))
          "Long"  (map long (stream-seq! (.doubleStream vv-column)))
          "String" (stream-seq! (.stream vv-column))
          "Char"  (map char (stream-seq! (.intStream vv-column)))
          "Byte" (stream-seq! (.intStream vv-column))
          "Short" (stream-seq! (.intStream vv-column))
          )
        ]
    (def vv-column vv-column)
    (if (contains? #{"Object","String"} vv-type)
      (ds/new-column col-name the-seq)
      
      (ds/new-column col-name the-seq {} 
                     (if (.isNullable vv-column)
                       (stream-seq! (.stream (.getNullMask vv-column)))
                       nil
                       )
                     ))))

(defn df->ds [df]
  (ds/new-dataset
   (map
    value-vector->column
    (.columns df))))


(defn- replace-nil [col replacement]
  (map
   #(if (nil? %)
      replacement
      %)
   col))

(defn- construct [klass & args]
  (clojure.lang.Reflector/invokeConstructor klass (into-array Object args)))

(def transformers
  {:int        [smile.data.vector.NullableIntVector int-array 0]
   :int16      [smile.data.vector.NullableShortVector short-array 0]
   :int64      [smile.data.vector.NullableLongVector  long-array 0]
   :float64    [smile.data.vector.NullableDoubleVector double-array 0] 
   :float32    [smile.data.vector.NullableFloatVector  float-array 0]
   :char       [smile.data.vector.NullableCharVector  char-array \0]
   :boolean    [smile.data.vector.NullableBooleanVector boolean-array false]
   :string     [smile.data.vector.StringVector nil nil]
   :int8       [smile.data.vector.NullableByteVector byte-array 0]
   })

(defn- col->value-vector [col]
  (let [bs (java.util.BitSet.)
        col-name (str (-> col meta :name))
        datatype (-> col meta :datatype)
        [clazz array-fn replacement] (get transformers datatype
                                          [smile.data.vector.ObjectVector nil nil])]
    (run!
     (fn [idx]
       (.set bs idx true))
     (ds/missing col))

    
    (if (nil? clazz)
      (throw (IllegalArgumentException. (format "datatype %s is not supported" datatype)))
      
      (case (str (.getName clazz))
        "smile.data.vector.ObjectVector" (ObjectVector. col-name (object-array col))
        "smile.data.vector.StringVector" (StringVector. col-name (into-array String col))


        (construct clazz
                   col-name
                   (array-fn (replace-nil col replacement))
                   bs)))))



(defn ds->df [ds]
  (DataFrame.
   (into-array ValueVector
               (map
                (fn [col]
                  (col->value-vector col))
                (ds/columns ds)))))









(comment
  (require '[smile.datasets CPU])
  (def iris
    (scicloj.metamorph.ml.toydata/iris-ds))
  


  (def cpu
    (.data (CPU.
            (.toPath
             (io/file "cpu.arff")))))
  

  (def cpu-with-null
    (.data (CPU.
            (.toPath
             (io/file "cpu_with_null.arff")))))
  

  (-> iris
      ds->df
      df->ds)
  

  (df->ds cpu)

  (df->ds cpu-with-null)
  )


