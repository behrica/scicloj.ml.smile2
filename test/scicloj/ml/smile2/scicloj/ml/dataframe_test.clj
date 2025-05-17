(ns scicloj.ml.smile2.scicloj.ml.dataframe-test 
  (:require
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml.toydata :as data]
   [scicloj.ml.smile2.dataframe :as dataframe]
   [tech.v3.dataset :as ds]))

(defn- validate-round-trip [ds]
  (let [round-tripped

        (-> ds
            (dataframe/ds->df)
            (dataframe/df->ds))]
    (is (= ds round-tripped))))



(deftest test-validate-round-trip

  (validate-round-trip
   (->
    (data/iris-ds)
    (ds/rename-columns ["sepal_length" "sepal_width" "petal_length" "petal_width" "species"])))

  (validate-round-trip
   (->
    (data/breast-cancer-ds)
    (ds/rename-columns
     ["mean-radius"
      "mean-texture"
      "mean-perimeter"
      "mean-area"
      "mean-smoothness"
      "mean-compactness"
      "mean-concavity"
      "mean-concave-points"
      "mean-symmetry"
      "mean-fractal-dimension"
      "radius-error"
      "texture-error"
      "perimeter-error"
      "area-error"
      "smoothness-error"
      "compactness-error"
      "concavity-error"
      "concave-points-error"
      "symmetry-error"
      "fractal-dimension-error"
      "worst-radius"
      "worst-texture"
      "worst-perimeter"
      "worst-area"
      "worst-smoothness"
      "worst-compactness"
      "worst-concavity"
      "worst-concave-points"
      "worst-symmetry"
      "worst-fractal-dimension"
      "class"])))

  (validate-round-trip
   (ds/->dataset {"a" [true false true]
                  "a1" [true nil true]
                  "b" ["x" nil "z"]
                  "c" [1 2 3]
                  "d" [1.0 nil 2.0]
                  "e" [0.1 0.2 0.3]
                  "f" [nil 0.2 nil]
                  "g" [nil nil nil]
                  "h" [1 "x" 1.0]
                  "i" [[1 2] [3 4] [5 6]]
                  "c1" [1 nil 3]
                  "j" ["a" "b" "c"]
                  "j1" ["a" "b" nil]
                  "k" [\a \b \c]
                  "k1" [\a \b nil]
                  "l" (byte-array (map byte [0 1 2]))
                  "m" (short-array (map short [0 1 2]))
                  "n" [:a :b :c]
                  "o" [{:foo 1} {:bar 2} {:baz 3}]

                  })))

(deftest specific-data
  (is
   (->
    (dataframe/ds->df (ds/->dataset {"k" [\a \b \c]}))
    .dtypes
    first
    .isChar))

  (is
   (->
    (dataframe/ds->df (ds/->dataset {"k" (byte-array (map byte [0 1 2]))}))
    .dtypes
    first
    .isByte))

  (is
   (->
    (dataframe/ds->df (ds/->dataset {"k" (short-array (map short [0 1 2]))}))
    .dtypes
    first
    .isShort)))




