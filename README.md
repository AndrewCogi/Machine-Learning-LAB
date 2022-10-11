# Machine-Learning-LAB

Perform Machine Learning using various models and analyzed for results.

<b>Index
1. LAB 1 Proposal
2. LAB 1 Conclusion
3. LAB 2 Proposal
4. LAB 2 Conclusion
</b>

# LAB 1 Proposal
### <data exploration & data preprocessing - without scaling & encoding>

The number of records is 699 ,but the unique value is 645. There is a record with overlapping IDs. Because the contents of the data in the overlapping ID are different, we do not remove the overlapping ID data. (We will drop ID feature)
Bare Muclei data contains missing data.('?')
-> So, Drop the whole feature OR Delete missing data only.

Q. Is there anything to drop for feature?

A. No. Even if it is concentrated on one side, it is because it is a disease search, so a small number of data can be important. Therefore, We think to save all data because a small number of data is likely to be important.


### <data preprocessing - scaling & encoding>

Scaling : We will use 5 different scaling methods. (MinMax, Robust, Standard, MaxAbs, Nomalizer)<br>
Encoding : dataset is all numerical type. So, Encoding is unnecessary.


### <Model building&gt;

Data split : test_size=0.3<br>
We use 4 modeling algorithms and each parameters.<br>
logistic regression : solver = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}<br>
DT (entropy) : criterion = entropy, max_features= {“auto”, “sqrt”, “log2”}<br>
DT (gini) : criterion = gini, max_features= {“auto”, “sqrt”, “log2”}<br>
SVC : kernel = {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, gamma = {‘scale’, ‘auto’}, C = {0.01, 0.1, 1}<br>
In addition, in the case of model instances, it is collected separately in the array for score calculation in testing.


### <testing&gt;
Calculate the score by adjusting the k(Cross validataion paramater 3,5,7) value with the model instances stored in the array. Also, print the top 5 accuracy model.


### <Flowchart of simple code&gt;
```python
def do_classification()
  load_and_modification_dataset()
  for scaler in [scalers]:
    Scaling()
    DataSplit() ← test_size = {0.1, 0.2, 0.3}
    for model in [models]:
      Modeling()
      newArr ← saving model instance
    CrossValidation() ← k={3,5,7}
  top_5 = findTop5()
  return top_5

main(){
  top_5 = do_classification()
  print(top_5)
}
```

# LAB 1 Conclusion

<b>Index
1. Conclusion
2. Team member contribution
</b>

## 1. Conclusion

First, we checked the null value for dataset. Based on the experience, “ID” is determined as a column that is not necessary in this data analysis. After checking the dataset, “Bare Nuclears” is numeric data, but the type is object type. Also we found that some of the value is '?'. To preserve as much data as possible, '?' were replaced with the mean value of the corresponding column. 

Use Correlation to check the relationship between the target feature "class" and the rest of the features. We find that “Mitoses” is relatively less associated with “class” than other features. So, we decided to drop "Mitoses" column.

![Corr Matrix](https://user-images.githubusercontent.com/69946205/194837456-2e6eeef9-bbd6-4f38-ab9d-9091a8e0a637.png)

When we print out all the results, we can see that svm has the highest accuracy. The next ranking is the logistic regression model, and the third is the decision tree model. Also, in svm, the kernel in the form of sigmoid was mainly high in accuracy.

Since linear svm is limited to two dimensions, it can be confirmed that the accuracy appears high in other types of kernels. And the difference in score for the criterion of the decision tree was very small.

C is a parameter that determines how many data samples are allowed to be placed in another class, and it can be seen that the C value shows high accuracy when it is 0.1. However, it should be noted that there is a risk of underfitting if the C value is too small and overfitting if it is high.(also in gamma). Also, in the SVM model, the score showed a big difference according to gamma and C parameters. (97%~30%)

In the case of the scaler, the standard scaler had the highest average value on average and the minmax was the lowest. (However, since this is an average that does not take into account other columns, it cannot be said to be an exact number, but it is a number that is somewhat consistent when directly checking the upper, middle, and lower ranks in accuracy. ) Also the score difference according to the scaling method was very small. <br>

![제목 없음](https://user-images.githubusercontent.com/69946205/194837945-ee350c16-0b1c-4cab-94ff-1419523efa1b.png)

SVM is a powerful model and works well on a variety of datasets. However, it should not always be concluded that SVM is powerful. When the amount of samples is large, it often does not fit well. After all, it is also up to the user to decide this.

## 2. Team member contribution

All team members wrote their own code from start to finish, had a meeting time to modify and integrate the code, and analyze it together.

![Contribution](https://user-images.githubusercontent.com/69946205/194838213-4a4e230a-36ee-4e1c-a974-85a304532aa0.png)

# LAB 2 Proposal

### <data exploration & data preprocessing (without scaling, encoding)&gt;
The dataset consists of a numerical value except for one feature. (median_house_value)<br>
And the total_bedrooms feature contains the NaN value. (207 records)<br>
-> Drop the dirty record(NaN).<br>
According to the professor's words, the median_house_value feature should be dropped.<br>
-> Drop 'median_house_value' feature.<br>
In addition, in order to process the Outlier, we need to plot the dataset and drop it for records that are more than(or less than) a specific value.<br>
-> Drop outliers in each numerical features.<br>
Make combinations of the features using PCA. <br>

### <data preprocessing - scaling & encoding>
Scaling : We will use 5 different scaling methods. (MinMax, Robust, Standard, MaxAbs, Nomalizer)<br>
Encoding : For ocean_proximity feature -> [ <1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN ] <br>
We will use 3 different encoding methods. (LabelEncoder, OneHotEncoder, OrdinalEncoder)

### <model building&gt;
We use 5 modeling algorithms and each parameters.<br>
K-means : n_clusters, algorithm : {“lloyd”, “elkan”, “auto”, “full”}<br>
EM(GMM) : n_components, init_params{‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’}<br>
CLARANS :	number_clusters, numlocal, maxneighbor<br>
DBSCAN : eps, metric : {'euclidean','manhattan'}, algorithm : {‘ball_tree’, ‘kd_tree’, ‘brute’}<br>
Spectral clustering : n_components, gamma, n_neighbors, eigen_solver : {‘arpack’, ‘lobpcg’, ‘amg’}<br>

### <testing&gt;
Through the silhouette score and elbow method, check how many clusters have high accuracy. Test how evenly the data is distributed inside the cluster with purity.

### <Flowchart of simple code&gt;
```python
def AutoML(scaler list, encoder list, model list, dataset, test_size)
  load_and_modification_dataset()
  for scaler in [scalers]:
    Scaling_Encoding()
    for model in [models]:
      Modeling()
      Testing_Plotting()

main(){
  // original dataset
  AutoML(...)
  // feature combination dataset
  AutoML(...)
}
```

# LAB 2 Conclusion


<b>Index
1. Conclusion
2. AutoML Description
3. Team member contribution
</b>

## 1. Conclusion

 As a result of checking the dataset as a whole, the null value was 207 null value in “total_bedrooms" feature. All records with null values were deleted because the ratio of null values was lower in comparison the entire dataset. The “median_house_value” feature was deleted because it is target feature. Encoding was performed because the "ocean_proximity" feature is a categorical values. We use two types of encoders: label encoder, ordinal encoder and five types of scalers: standard scaler, minmax scaler, robust scaler, maxabs scaler, normalizer. We use five types of model: KMeans, GMM, CLARANS, DBSCAN, Mean Shift.

 IQR method was used to remove outliers from the original data. The percentages were 15 and 85, respectively, and about 1000 rows were dropped. However, it does not seem to show a significant performance difference, so we did not add it to the code.
 
![iqr](https://user-images.githubusercontent.com/74485630/195056012-9a2415e4-a874-4eea-8f79-b117ee5b6f87.png)

AutoML was run with all data and all parameters. The difference between encoder and scaler was very small. As a result of analyzing the entire data, it was found that clustering did not proceed well because there were a total of 9 features. In fact, as a result of visually checking through plotting, there was no model in which all feature-pairs were perfectly grouped.

### DBSCAN
 In DBSCAN, the values for eps, min_samples, and metric were changed. The parameter that showed the greatest influence was eps. When the range to be included in the cluster was set small, we could see a situation in which all data were identified as outliers. Also, if the eps is too high, most of the data is meaninglessly bundled. We also saw the change according to the number of min_samples at the same eps value. Good results were obtained when the number of min_samples was appropriate and eps was also at an appropriate value. When the eps value was between 1 and 1.5 and min-samples were between 100 and 200, it was judged that they were well bound. In addition, the distance method did not seem to have a significant difference visually.
 
![dbscan](https://user-images.githubusercontent.com/74485630/195056932-e3ad2919-1770-47ae-93c8-fe8869efaa3b.png)

### GMM
 As a result of calculating the Silhouette score for GMM, the euclidean method and the manhattan method showed a similar flow. When modeling for GMM, if parameters are set incorrectly, most records are included in one cluster. Also, the average value for medianHouseValue was similar to a meaningless level. However, for good parameters(covariance_type=tied,init_params=random, n_components=2,3), it was possible to see the overall appearance of being divided into two categories, and 10,000 records were grouped in a cluster, and the medianHouseValue value was also more than 20,000, showing a clear difference. It was found out how much the parameter setting affects the clustering result. Except for different cases for each combination of columns, it seems that gmm was properly grouped in each cluster and had good performance when judging the overall model.
 
![GMM](https://user-images.githubusercontent.com/74485630/195057000-8b9a5cf4-5079-48a4-8a8e-f98a9f21e9f7.png)

### KMeans
 In the case of KMeans, we saw a result that shows good results when n_cluster is 2,5 through the Silhouette score. In fact, as shown in the figure, the best result was shown at 5. Also, we tried calculating the silhouette score for the euclidean method and the manhattan method, respectively, and it showed a similar graph shape overall. (Clustering was carried out for the commonly used euclidean method) In particular, it was found that the higher the value of max_iter, the denser the clustering proceeded. We analyzed how many medianHouseValue, the existing target value, were grouped for the clustering result, how much are the maximum and minimum values, and how much is the average value. Even in the results that clustering results seem to be well bound, only about 10000 differences were seen with respect to the average value of medianHouseValue, and no significant differences were seen. I think this result appears because the number of features is too large, and the computer is mathematically dependent only on the data.
 
![Kmeans](https://user-images.githubusercontent.com/74485630/195057054-7cf096a5-597b-4991-9ca4-7e332a1447f0.png)

### CLARANS
 For CLARANS, the number of clusters, numlocal, and maxneighbor parameters were entered and modeled. (numberlocal represents the number of local minima obtained and maxneighbor represents the maximum number of neighbors) It was not included in the sklearn, so it proceeded to the pyclustering module. The output result is a two-dimensional reduction of all features. Most clustering results usually showed good clustering results.
 
### Mean Shift
 Finally, modeling was conducted on Mean Shift, the subject of Active Learning. As a parameter, I set how generously to include in the cluster by changing the bandwidth value. Also, using estimate_bandwidth, we found the optimal bandwidth value. (The result was about 3) we saw that the smaller the bandwidth, the more clusters are created. Conversely, the larger the bandwidth, the fewer clusters were created. As for the results, most of the values were concentrated for cluster label=0, and in conclusion, clustering was not done properly.
 
![meanshift](https://user-images.githubusercontent.com/74485630/195057107-3b942187-362c-4248-a7c5-96953a843c70.png)

### Result
 As a result of analyzing the clustering results for all models, a common shape was found, but most clustering was not good for longitude, latitude, medianIncome, and housingMedianAge. Therefore, considering these features as features that do not affect clustering, we excluded them and ran AutoML again.
 For all data, it was possible to see the results of clustering more reliably than before, and it was also seen that the houseMedianValue also differed more for each cluster. Therefore, I found that it is important to think about and decide whether a feature will be helpful for clustering or not. Even if the dataset is good, invest a lot of time and analyze it to set the appropriate model and appropriate parameter to set the optimal value. I found it important and difficult to find.
 
## 2. AutoML Description

![automl](https://user-images.githubusercontent.com/74485630/195065782-6f1e9f8c-bc02-4b47-b0cc-92e8616296bc.png)

## 3. Team member contribution

All team members wrote their own code from start to finish, had a meeting time to modify and integrate the code, and analyze it together.

![team](https://user-images.githubusercontent.com/74485630/195057294-e94a0c11-71d3-4190-afc1-7a450d9e4905.png)
