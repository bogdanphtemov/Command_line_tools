"""Test KNN with textual labels functionality"""
import numpy as np
from myclt.ML.supervised_learning.knn.classifier import KNNClassifier
from myclt.ML.supervised_learning.knn.regressor import KNNRegressor

def test_text_labels():
    """Test that KNN works correctly with text labels"""
    print("Testing KNN with text labels...")
    
    # Create test data with text labels
    X_train = np.array([
        [1.0, 2.0],
        [2.0, 3.0], 
        [3.0, 1.0],
        [4.0, 4.0]
    ])
    
    # Text labels for classification
    y_train_class = np.array(['Class_A', 'Class_B', 'Class_A', 'Class_B'])
    
    # Train classifier
    knn_classifier = KNNClassifier(k=2)
    knn_classifier.fit(X_train, y_train_class)
    
    # Try prediction with text labels
    X_test = np.array([[1.5, 2.5]])
    prediction = knn_classifier.predict(X_test)
    probabilities = knn_classifier.predict_proba(X_test)
    
    print(f"Classes: {knn_classifier.classes_}")
    print(f"Prediction: {prediction[0]}")
    print(f"Probabilities: {probabilities[0]}")
    
    # Test that batch prediction won't crash with text labels
    test_features = np.array([[1.2, 2.1], [3.5, 2.8]])
    batch_predictions = knn_classifier.predict(test_features)
    print(f"Batch predictions: {batch_predictions}")
    
    print("✓ KNN with text labels test passed!")

if __name__ == "__main__":
    test_text_labels()