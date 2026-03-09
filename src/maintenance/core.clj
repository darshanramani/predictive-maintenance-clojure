(ns maintenance.core
  (:gen-class)
  (:import
   [java.io File]
   [java.util Random]
   [weka.core Instances]
   [weka.core.converters CSVLoader]
   [weka.filters Filter]
   [weka.filters.unsupervised.attribute Remove]
   [weka.filters.unsupervised.attribute StringToNominal]
   [weka.filters.unsupervised.attribute NumericToNominal]
   [weka.classifiers Evaluation]
   [weka.classifiers.trees J48 RandomForest]))

(defn load-instances-from-csv [csv-path]
  (let [loader (CSVLoader.)]
    (.setSource loader (File. csv-path))
    (.getDataSet loader)))

(defn apply-remove-filter [^Instances data remove-range]
  (let [f (Remove.)]
    (.setAttributeIndices f remove-range)
    (.setInvertSelection f false)
    (.setInputFormat f data)
    (Filter/useFilter data f)))

(defn apply-string-to-nominal [^Instances data attr-range]
  (let [f (StringToNominal.)]
    (.setAttributeRange f attr-range)
    (.setInputFormat f data)
    (Filter/useFilter data f)))

(defn apply-numeric-to-nominal [^Instances data attr-range]
  (let [f (NumericToNominal.)]
    (.setAttributeIndices f attr-range)
    (.setInputFormat f data)
    (Filter/useFilter data f)))

(defn accuracy [^Evaluation eval]
  (* 100.0 (- 1.0 (.errorRate eval))))

(defn evaluate-model [model-name model data]
  (let [eval (Evaluation. data)
        rand (Random. 42)]
    (.crossValidateModel eval model data 10 rand)
    (println "----------------------------------")
    (println model-name "using 10-fold Cross Validation")
    (println "Accuracy (%):" (format "%.2f" (accuracy eval)))
    (println "Confusion Matrix:")
    (doseq [row (.confusionMatrix eval)]
      (println (vec row)))
    (println "----------------------------------")
    (accuracy eval)))

(defn -main [& _args]
  (println "Loading dataset...")
  (let [raw (load-instances-from-csv "data/ai4i2020.csv")]
    (println "Total rows loaded:" (.numInstances raw))
    (println "Total columns:" (.numAttributes raw))
    (let [step1 (apply-remove-filter raw "1-2,10-14")
          step2 (apply-string-to-nominal step1 "1")
          step3 (apply-numeric-to-nominal step2 "7")
          _ (.setClassIndex step3 (dec (.numAttributes step3)))
          decision-tree (J48.)
          random-forest (RandomForest.)
          dt-acc (evaluate-model "Decision Tree (J48)" decision-tree step3)
          rf-acc (evaluate-model "Random Forest" random-forest step3)]
      (.buildClassifier decision-tree step3)
      (println "Decision Tree Structure:")
      (println decision-tree)
      (println "Final Model Comparison")
      (println "Decision Tree Accuracy (%):" (format "%.2f" dt-acc))
      (println "Random Forest Accuracy (%):" (format "%.2f" rf-acc))
      (if (> rf-acc dt-acc)
        (println "Random Forest performed better than Decision Tree.")
        (if (< rf-acc dt-acc)
          (println "Decision Tree performed better than Random Forest.")
          (println "Both models performed equally.")))
      (println "Done"))))