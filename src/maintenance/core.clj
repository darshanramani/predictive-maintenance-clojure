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
    [weka.classifiers.trees J48]
    [weka.classifiers Evaluation]))

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
    (.setAttributeIndices f attr-range) ; e.g. "7"
    (.setInputFormat f data)
    (Filter/useFilter data f)))

(defn accuracy [^Evaluation eval]
  (* 100.0 (- 1.0 (.errorRate eval))))

(defn -main [& _args]
  (println "Loading dataset...")
  (let [raw (load-instances-from-csv "data/ai4i2020.csv")]
    (println "Total rows loaded:" (.numInstances raw))
    (println "Total columns:" (.numAttributes raw))

    ;; Remove leakage columns and IDs
    (let [step1 (apply-remove-filter raw "1-2,10-14")
          ;; Now: 1 Type, 2 AirTemp, 3 ProcTemp, 4 RotSpeed, 5 Torque, 6 ToolWear, 7 Machine failure
          step2 (apply-string-to-nominal step1 "1")   ; Type -> nominal
          step3 (apply-numeric-to-nominal step2 "7")  ; Machine failure -> nominal
          _ (.setClassIndex step3 (dec (.numAttributes step3)))

          model (J48.)
          eval (Evaluation. step3)
          rand (Random. 42)]

      ;; 10-fold cross validation
      (.crossValidateModel eval model step3 10 rand)

      (println "----------------------------------")
      (println "Decision Tree (J48) using 10-fold Cross Validation")
      (println "Accuracy (%):" (format "%.2f" (accuracy eval)))
      (println "Confusion Matrix:")
      (doseq [row (.confusionMatrix eval)]
        (println (vec row)))
      (println "----------------------------------")
      (println "Done"))))
