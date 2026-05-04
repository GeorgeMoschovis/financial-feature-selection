import time
from evaluate_features import Evaluator
from evaluate_features_rf import EvaluatorRF
from tabuSearch import tabu_search_feature_selection

# Choose classifier: "knn" or "rf"
CLASSIFIER = "knn"

if CLASSIFIER == "knn":
    evaluator = Evaluator(k=5, cv_folds=5, sample_size=1000, seed=42)
elif CLASSIFIER == "rf":
    # n_estimators=20 keeps RF fast enough for the Tabu loop
    evaluator = EvaluatorRF(cv_folds=3, sample_size=500, n_estimators=20, seed=42)

t0 = time.time()
best_subset, best_score, history = tabu_search_feature_selection(
    n_features=evaluator.num_features,
    evaluate=evaluator.evaluate,
    max_iter=100,
    tabu_size=10,
    seed=42,
)
elapsed = time.time() - t0

report = evaluator.final_report(best_subset)

print(f"\nClassifier: {CLASSIFIER.upper()}")
print(f"Best OCA  : {report['OCA']:.4f} ({report['OCA']*100:.2f}%)")
print(f"AUC       : {report['AUC']:.4f}")
print(f"ACA       : {report['ACA']:.4f}")
print(f"RMSE      : {report['RMSE']:.4f}")
print(f"Per-class : class 0 (alive) = {report['per_class_accuracy'][0]:.4f}  |  class 1 (failed) = {report['per_class_accuracy'][1]:.4f}")
print(f"Features selected ({report['n_features']}/{evaluator.num_features}):")
for i in sorted(best_subset):
    print(f"  [{i:2d}] {evaluator.feature_names[i]}")

print(f"\nIterations run    : {len(history)}")
print(f"Unique subsets    : {evaluator.cache_size}")
print(f"Elapsed           : {elapsed:.1f}s")
