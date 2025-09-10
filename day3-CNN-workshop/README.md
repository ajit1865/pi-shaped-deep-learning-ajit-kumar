# Day 3: Convolutional Neural Networks (CNN)

## Objective
### Build and train a Convolutional Neural Network (CNN) on the Fashion-MNIST dataset and evaluate its performance.


## 1. What advantages do CNNs have over traditional fully connected neural networks for image data?

Convolutional Neural Networks (CNNs) are better suited for image data because they preserve spatial relationships between pixels, unlike fully connected networks that flatten inputs. CNNs use shared weights through filters, reducing the number of parameters and computational cost. They also automatically learn hierarchical features like edges, textures, and shapes, which improves accuracy, scalability, and generalization for image classification, object detection, and computer vision tasks compared to traditional dense architectures.

## 2. What is the role of convolutional filters/kernels in a CNN?

Convolutional filters (kernels) are small matrices that slide over the input image to detect local features such as edges, corners, or textures. Each filter learns to capture a different pattern during training. By stacking multiple filters and layers, CNNs can detect increasingly complex features. Filters reduce parameter count, enforce locality, and make CNNs translation-invariant, allowing them to recognize objects regardless of their position within the input image.

## 3. Why do we use pooling layers, and what is the difference between MaxPooling and AveragePooling?

Pooling layers reduce the spatial dimensions of feature maps, lowering computation and preventing overfitting while keeping essential features. MaxPooling selects the maximum value from a region, preserving strong activations (edges/features). AveragePooling computes the average of values, providing smoother representations but may lose detail. MaxPooling is often preferred in practice as it emphasizes prominent features, making models more robust to noise and variations in the input image.

## 4. Why is normalization of image pixels important before training?

Normalization of image pixels ensures that input values fall within a consistent range, usually [0,1] or standardized (mean 0, variance 1). This helps neural networks train faster and more effectively by stabilizing gradients, preventing vanishing/exploding values, and improving convergence. Without normalization, large pixel intensity differences can dominate weight updates, leading to unstable training. It also ensures fair treatment of all features in the learning process.

## 5. How does the softmax activation function work in multi-class classification?

The softmax activation function converts raw outputs (logits) of a neural network into probabilities for each class. It exponentiates each logit, then divides by the sum of all exponentials, ensuring values fall between 0 and 1 and sum to 1. This makes it easy to interpret the model’s prediction as a probability distribution. The highest probability indicates the predicted class, making softmax essential for multi-class classification tasks.

## 6. What strategies can help prevent overfitting in CNNs? (e.g., dropout, data augmentation)

Overfitting occurs when a CNN memorizes training data rather than generalizing. Strategies include dropout, which randomly disables neurons during training to encourage robustness; data augmentation, which generates new training samples via transformations (rotation, flipping, scaling); and regularization techniques like L2 weight decay. Early stopping, smaller architectures, and batch normalization also help. These strategies ensure the CNN generalizes better, improving performance on unseen test data.

## 7. What does the confusion matrix tell you about model performance?

A confusion matrix provides detailed insights beyond accuracy by showing the distribution of true positives, true negatives, false positives, and false negatives. It highlights where the model is making errors, such as misclassifying one class as another. From it, metrics like precision, recall, and F1-score can be calculated. This allows deeper understanding of class-level performance, imbalance handling, and areas where the model struggles most in predictions.

## 8. If you wanted to improve the CNN, what architectural or data changes would you try?

To improve a CNN, you could experiment with deeper architectures, adding more convolutional and pooling layers to capture richer features. Using advanced architectures like ResNet or DenseNet helps avoid vanishing gradients. From the data side, applying more augmentation, balancing datasets, or including additional features (like color channels or contextual information) can improve performance. Hyperparameter tuning, transfer learning with pre-trained models, and regularization techniques also enhance results.
