# MLEM

gang_distribution_report: creates graphs for final reports (shows, for example, gangs with only one row (these must be removed cause they cannot be stratified between training, test and validation set) )

normalized_dataset: creates a dataset with no dirty data
dataset_ML_formatter: applies a one hot encoding
stratifiaction_dataset: a script that calculates the stratification of the dataset (this means: keep same distribution of gangs between train/test/validation set)


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train_enc = smote.fit_resample(X_train, y_train_enc)

- Non migliora generando dati fittizi

- Non serve normalizzare perché sono tutti binari

- Confronto con metodo grid search: tecnica ottimizzazione iperparametri (ogni modello ha suoi parametri e sono da impostare pretraining). Si crea un array di valori e il grid search testa tutte le possibili combinazioni di parametri triandoti fuori i parametri del modello migliore


In model_comparison_results.csv sono presenti le analisi di tutti i modelli a seguito di ottimizzazione. XGBoost è il migliore.

