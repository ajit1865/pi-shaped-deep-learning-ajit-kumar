# pi-shaped-deep-learning-ajit-kumar



---

## Day 1 - Neural Networks Workshop  

This repo contains an implementation of a feedforward neural network to classify breast cancer tumors (malignant vs benign) using the sklearn Breast Cancer dataset.  

---
# Core Concept Questions & Answers

## 1. What is the role of feature scaling/normalization in training neural networks?
#### Feature scaling or normalization is important because neural networks are highly sensitive to the scale of input features. If features are on very different scales (e.g., age in years vs. salary in dollars), the optimization process may take much longer or even fail to converge properly. By scaling features to a similar range (like 0–1 or standardized with mean 0 and variance 1), we ensure that gradients update weights more uniformly, which speeds up convergence, reduces the risk of exploding/vanishing gradients, and improves training stability.

## 2. Why do we split data into training and testing sets?
#### The primary reason for splitting the dataset is to evaluate the model’s ability to generalize. The training set is used to fit the model and adjust parameters, while the test set provides unseen data to check how well the model performs on real-world scenarios. If we only trained and evaluated on the same data, the model might simply memorize the data (overfit), giving an unrealistic estimate of performance. The test set simulates new data and gives a reliable measure of generalization.

## 3. What is the purpose of activation functions like ReLU or Sigmoid?
#### Without activation functions, neural networks would just be linear models regardless of the number of layers. Activation functions add non-linearity, which allows networks to approximate complex functions.

#### ReLU (Rectified Linear Unit): Outputs zero for negative values and the value itself for positive inputs. It helps reduce vanishing gradient problems and trains faster in deep models.

#### Sigmoid: Squashes values between 0 and 1, making it suitable for representing probabilities in binary classification. However, it can suffer from vanishing gradients.
Thus, activation functions allow networks to capture complex, non-linear relationships in the data.

## 4. Why is binary cross-entropy commonly used as a loss function for classification?
#### Binary cross-entropy compares predicted probabilities with the actual binary labels and penalizes incorrect predictions. It is particularly effective because it strongly penalizes confident but wrong predictions, pushing the model to output probabilities that better reflect uncertainty. Unlike mean squared error, which isn’t ideal for classification, binary cross-entropy directly measures the difference in probability distributions, making it the natural choice for binary classification problems.

## 5. How does the optimizer (e.g., Adam) affect training compared to plain gradient descent?
#### Plain gradient descent updates weights in the direction of the gradient with a fixed learning rate, which can be slow and inefficient. Optimizers like Adam (Adaptive Moment Estimation) improve upon this by maintaining adaptive learning rates for each parameter based on both the first moment (mean of gradients) and second moment (variance of gradients). This leads to faster convergence, better handling of sparse data, and generally improved training stability compared to standard gradient descent.

## 6. What does the confusion matrix tell you beyond just accuracy?
#### Accuracy only measures overall correctness but does not reveal the type of errors the model is making. The confusion matrix breaks predictions down into:

#### True Positives (TP) – correctly predicted positives

#### True Negatives (TN) – correctly predicted negatives

#### False Positives (FP) – incorrectly predicted positives

#### False Negatives (FN) – incorrectly predicted negatives
#### By analyzing these, we can detect if the model is biased towards one class, or if it’s missing important positive cases, which is especially critical in fields like healthcare or fraud detection.

## 7. How can increasing the number of hidden layers or neurons impact model performance?
#### Adding more layers or neurons increases the model’s capacity to learn complex functions and capture more detailed patterns in the data. However, this comes at a cost:

#### Pros: Better representation of complex data, potentially higher accuracy.

#### Cons: Higher risk of overfitting, longer training times, and increased computational costs.
So while deeper or wider networks may improve performance, they require careful tuning, regularization, and sufficient training data to avoid overfitting.

## 8. What are some signs that your model is overfitting the training data?
Overfitting happens when the model learns the training data too well, including noise and outliers, and fails to generalize to unseen data. Signs include:

#### Very high training accuracy but much lower test accuracy.

#### The validation loss stops decreasing while training loss continues to drop.

#### The model performs poorly on new or real-world data.
#### This indicates that the model is memorizing patterns instead of learning general features.

## 9. Why do we evaluate using precision, recall, and F1-score instead of accuracy alone?
#### Accuracy can be misleading, especially with imbalanced datasets (e.g., detecting rare diseases where 95% of samples are negative). Precision and recall provide a clearer picture:

#### Precision: Of all predicted positives, how many are correct.

#### Recall: Of all actual positives, how many were identified correctly.

#### F1-score: Harmonic mean of precision and recall, useful for balancing the two.
These metrics give a more complete view of model performance, especially when the cost of false positives and false negatives differs.

## 10. How would you improve the model if it performs poorly on the test set?
#### If the model does not perform well on unseen data, several strategies can be applied:

#### Collect more or better-quality data to help the model learn general patterns.

#### Improve preprocessing such as scaling, feature engineering, or handling missing values.

#### Apply regularization methods like Dropout or L2 weight penalties to reduce overfitting.

#### Hyperparameter tuning, adjusting learning rate, batch size, and architecture depth/width.

#### Experiment with different models or architectures, such as deeper networks, CNNs, or ensemble methods.

#### Data augmentation (in images or text) to make the model robust.
#### This iterative process ensures that the model improves its generalization ability.
