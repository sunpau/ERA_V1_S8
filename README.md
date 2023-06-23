# ERA_V1_S8
The objective of the assignment is to observe the effect of different kind of Normalization using CIFAR 10 dataset

- Loss using different Normalization
  -  For batch size = 64, and keeping all the hyperparameters same,it is observed the Batch Normalization gives best accuracy for 20 epochs.
  -  The performance of Group and Layer Norm is almost comaprable and loss is converging by 20 epochs.
  ![lossvsnorm](./Images/LossVsNorm.png)
  

-  Batch Normalization
   -  Best Train Accuracy: 79.91%   Best Test Accuracy: 77.58%
   -  Summary
      ![batchsummary](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/BatchNormSummary.png)

   -  Loss and Accuracy
      ![batchloss](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/BatchNormLossAccuracy.png)

   -  Misclassified Images
      ![batchmisclassification](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/BatchNormMisClassification.png)

-  Layer Normalization
   -  Best Train Accuracy: 76.45%   Best Test Accuracy: 73.58%
   -  Summary
      ![layersummary](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/LayerNormSummary.png)

   -  Loss and Accuracy
      ![layerloss](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/LayerNormLossAccuracy.png)

   -  Misclassified Images
      ![layermisclassification](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/LayerNormMisclassification.png)

-  Group Normalization - used Group Size: 8
   -  Best Train Accuracy: 76.29%   Best Test Accuracy: 73.96%
   -  Summary
      ![groupsummary](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/GroupNormSummary.png)

   -  Loss and Accuracy
      ![grouploss](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/GroupNormLossAccuracy.png)

   -  Misclassified Images
      ![groupmisclassification](https://github.com/sunpau/ERA_V1_S8/blob/main/Images/GroupNormMisclassification.png)
      
      




