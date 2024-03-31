# FinalProject_TM

## Dataset
In order to be able to run this code you will need the next datasets for sentiment and topic analysis:
Sport https://www.kaggle.com/datasets/noahx1/ea-sports-fc-24-steam-reviews
Book - https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews
Movie - https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

And for the NERC we use 
Google dataset pre-trained word embeddings called word2vec.The embeddings have 300 dimensions.: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?pli=1&resourcekey=0-wjGZdNAUop6WykTtMip30g

## Coding description
### 1. NERC
The purpose of this notebook is Named Entity Recognition (NER), a crucial NLP job. The actions consist of:

- Loading Training Data: To prepare training data, use `ConllCorpusReader` from the `nltk` library.
- Preparing Test Data: Extracting test features and labels from a TSV file, then modifying individual labels to ensure uniformity.
- Model Application: (shown partially) involves predicting NER labels on the test data using word embeddings and a linear classifier, then comparing the predictions to the genuine labels.

From the classification report you provided for the 'NERC.ipynb', which seems to be the result of a Named Entity Recognition (NER) task, we can deduce the following:
  #### Results 
Overall Performance : The overall accuracy of the model on this test set is 0.86, which seems quite high. However, this metric is less informative for NER tasks, especially if the majority of tokens are 'O' (Outside of Named Entity), which seems to be the case given the 'support' of 160 for 'O' and an F1-score of 0.95.
  
-  Class-specific Performance :
  -  B-ORG  (Beginning of an Organization entity): The model has moderate precision (0.50) and recall (0.67), with an F1-score of 0.57. This suggests it can somewhat reliably identify the beginning of organization entities but still misses some and makes some false identifications.
  -  B-PER  (Beginning of a Person's name): Here, the model's precision is slightly higher than recall (0.60 vs. 0.50), with an F1-score of 0.55. It is better at identifying person names than organizations but not by a large margin.
  -  I-PER  (Inside a Person's name): For tokens that are inside a person's name, precision is equal to the recall (0.50), with an F1-score of 0.40, indicating a balanced but overall low performance for multi-token person names.
  
-  Low/Zero Scores : A significant number of entity types, such as B-DATE, B-MISC, I-DATE, I-LOC, and others, have zero precision, recall, and F1-scores. This could be because:
  - The model did not predict any instances of these classes (hence zero precision and F1-score).
  - The model failed to correctly identify any true instance of these classes that were present in the test set (hence zero recall).
  
-  Macro vs. Weighted Average : The macro average, which treats all classes equally, is quite low (0.21 across precision, recall, and F1-score), indicating that the model is not performing well across all classes. The weighted average is much higher (0.78 precision, 0.86 recall, 0.81 F1-score), which is influenced by the high support for 'O' tokens and suggests that the model's performance is skewed by the most represented class ('O').

-  Potential Data Imbalance : The 'support' column, which indicates the number of true instances for each class, shows that some classes like 'B-DATE', 'I-DATE', 'I-LOC', 'B-MISC', and 'I-WORK_OF_ART' have very few instances. This could mean that the dataset is imbalanced, which might be causing the model to struggle with these less represented classes.

-  Labeling Consistency : The presence of a label "BIO NER tag" with a support of 1 suggests there might be a consistency issue in the dataset or during the data preparation process.
    #### Future work
Class-specific Tuning : Given that the model performs well for some classes and poorly for others, consider implementing class-weighted training or oversampling techniques to address the imbalance.
-  Data Verification : Check the dataset for consistency, especially the 'BIO NER tag' and other classes with zero scores, to ensure there are no labeling errors.
-  Model Re-evaluation : If the dataset is heavily imbalanced with many 'O' tokens, consider using metrics like the macro-average F1-score for a more realistic evaluation of model performance across the different named entity classes.
-  Further Analysis : Examine false positives and false negatives in more detail to understand where and why the model is making errors. This could provide insight into whether the features being used are sufficient or if there's something specific about the data that's causing the model to struggle.



From the performance of Named Entity Recognition and Classification (NERC), we can derive several key insights: 1. Model Performance on Different Entity Types : - The model is very accurate in classifying non-entity tokens (labeled 'O'), which usually make up the majority of tokens in a dataset, as indicated by the high precision and recall (0.90 and 1.00, respectively) for this class. - For some specific entity types, such as 'B-ORG' (Beginning of Organization), 'B-PER' (Beginning of Person's name), and 'I-PER' (Inside a Person's name), the model is showing moderate performance with non-zero precision and recall values. - However, for many entity types like 'B-DATE', 'I-DATE', 'B-MISC', 'B-WORK_OF_ART', 'I-ORG', 'I-LOC', 'I-WORK_OF_ART', etc., the model fails to correctly predict any entity (precision, recall, and F1-score are 0.00). This indicates that the model is unable to recognize these entities in the text, or they may be underrepresented in the training data. 2. Imbalanced Dataset Indication : - The significant support for the 'O' class compared to other entities suggests that the dataset is highly imbalanced. It's common in NER tasks as the majority of the text typically does not contain named entities. 3. Confusion Matrix Analysis : - The confusion matrix shows that the model is heavily biased towards predicting the 'O' class, as indicated by the high number of true positives for 'O' and zeros for most other cells. - The off-diagonal elements in the confusion matrix, which would indicate misclassifications between different entity types, are mostly zeros, suggesting that the model rarely confuses between entity types; however, this could also be due to the lack of diversity in the predicted entities. 4. Precision, Recall, and F1-Score per Class Visualization : - The bar chart visualizing precision, recall, and F1-score per class reinforces the points above, showing that only a few classes have scores above zero. - There's a clear indication that while the model performs well for the 'O' class and a couple of entity types, it struggles with most others. This could be due to a variety of factors including data sparsity, class imbalance, lack of context, or features not being indicative enough for those entity types. 5. Potential for Future Work in Order to improve the model : - To improve the model's performance on entity types with poor metrics, one could consider gathering more training data for those classes, employing techniques like data augmentation, or using a more complex model that can capture context better, such as a BiLSTM or a Transformer-based model. - One might also need to review the annotation process to ensure consistency and correctness, as suggested by the presence of 'BIO NER tag' as an entity type, which seems like a data preprocessing error. The overall takeaway is that the NER model is effective at identifying non-entity portions of the text but needs improvements to accurately identify and classify the actual named entities.




### 2. sentiment Analysis.ipynb
Sentiment analysis and subject classification are the main objectives of this notebook, which focuses on reviews of movies, sports, and novels. The procedure consists of:

Importing libraries is necessary for machine learning and data processing.
- Data preprocessing and cleaning: combining datasets from three sources into a single format, handling missing values, and removing HTML tags from movie reviews.
- Sentiment Analysis and Topic Classification: To forecast sentiments and categorize the texts, NLP approaches and a machine learning model (`roberta-base} from `simpletransformers`) are used.
- Visualization and Evaluation: Using the test dataset, visualize the model's predictions' F1-scores, precision, and recall.

How we approached setniment analysis: 
1.   Function Definition (`train_data`)  : This function takes two parameters: `min_val` and `train_setting`.
   - `min_val` is used to set the minimum document frequency for terms in the `CountVectorizer`, which determines the minimum number of documents a term must appear in to be included in the vocabulary.
   - `train_setting` determines whether to use raw term frequencies (`'airline_count'`) or TF-IDF scores (`'tfidf'`) for training the classifier.

2.   Data Preprocessing  :
   - The function assumes the existence of a pandas DataFrame named `sentence_df` that contains sentences and their corresponding sentiments.
   - Sentences are converted to strings (in case they are not already).
   - The `CountVectorizer` is used to tokenize the text into words, remove English stopwords, and transform the text into a document-term matrix using the specified `min_val`.
   
3.   Feature Extraction  :
   - Depending on the `train_setting`, the document-term matrix is either kept as raw term frequencies or transformed into TF-IDF values using `TfidfTransformer`.

4.   Model Training and Splitting  :
   - The dataset is split into training and test sets with an 80-20 split.
   - A Multinomial Naive Bayes classifier is trained on the training set.

5.   Model Evaluation  :
   - The classifier is used to predict sentiments on the test set.
   - A classification report is generated, which includes precision, recall, and F1-score for each sentiment label.

6.   Experimentation Loop  :
   - The code iterates over two different `train_setting` values and three different `min_df` values to conduct experiments.
   - For each combination of settings, the model is trained and evaluated, and the results (classification reports) are printed out.

The code is set up for a series of experiments to see how the choice of feature representation (TF-IDF vs. raw counts) and the threshold for term frequency (minimum document frequency) affect the performance of a Multinomial Naive Bayes classifier on a text classification task. 
    #### Results
1.   Stable Performance Across `min_df` Values  : For both models, changes in the minimum document frequency (`min_df`) don't appear to significantly impact precision, recall, or F1-score. This stability suggests that the features (words) occurring at least 2 times are likely also present at higher thresholds, and they are informative enough for the classifier.

2.   Model Comparison  :
   - The "raw_term_freq" model consistently outperforms the "tfidf" model across all metrics and `min_df` settings. This could indicate that the raw frequency counts ("raw_term_freq" suggests a bag-of-words model using counts) are more effective for this particular dataset than TF-IDF weighted features.
   - The higher scores in the "raw_term_freq" model may also suggest that this model better captures the nuances of the text that are pertinent for classification, or the dataset is such that term frequency is a strong signal for sentiment classification.

3.   Best Settings for Each Model  :
   - For the "tfidf" model, increasing `min_df` from 2 to 5 improves recall and F1-score but does not change precision. Further increasing `min_df` to 10 doesn't significantly affect the scores.
   - The "raw_term_freq" model sees slight improvements in recall and F1-score with an increase in `min_df`, while precision remains relatively constant.

4.   Precision vs. Recall Trade-off  :
   - For the "tfidf" model, we notice that the precision is consistently higher than recall. This indicates that when the "tfidf" model predicts a sentiment, it is often correct, but it misses a larger number of actual instances (lower recall).
   - The "raw_term_freq" model, however, has a better balance between precision and recall, especially as `min_df` increases.

5.   F1-Score as a Harmonic Mean  :
   - The F1-score, which is the harmonic mean of precision and recall, is typically used to understand a model's overall effectiveness when there's a trade-off between precision and recall. In this visualization, the F1-score effectively captures the trends of precision and recall, offering a single metric to gauge performance.

From this visual representation, stakeholders can conclude that the "raw_term_freq" model with a higher `min_df` (5 or 10) may be the best performer for this dataset and task. However, it is important to consider other factors such as dataset balance, the significance of false positives vs. false negatives, and the potential impact of rare but informative words that might be excluded by a higher `min_df` threshold.

### 3. LDA and Linear Regression

#### Linear classification with cross-validation 

1. Performance Across Categories:
   - The 'sports' category has substantially higher 'support' than 'book' or 'movie'. This suggests that the 'sports' category has more samples in the dataset, which could potentially skew the classifier's performance towards this category.
   - The 'precision' and 'recall' for 'book' are quite low, indicating that the classifier has difficulty correctly identifying and classifying samples from the 'book' category.
   - The 'movie' category appears to have moderate 'precision' and 'recall', suggesting that the classifier's ability to correctly identify and classify 'movie' samples is better than 'book' but not as good as 'sports'.

2. Model Bias Toward Dominant Class:
   - Given the high 'support' for 'sports', the classifier might be biased toward this dominant class, possibly at the expense of the other categories.

3. Imbalanced Dataset Impact:
   - The imbalance in the number of samples for each category likely affects the classifier's learning, as it may not have enough 'book' examples to learn from effectively.

4. Evaluation Metric Considerations:
   - The 'f1-score' is a harmonic mean of 'precision' and 'recall', and it is especially low for 'book', indicating that the classifier is not performing well on this category. The 'f1-score' for 'sports' is not visible, which suggests that it might be off the scale used in the graph or not calculated correctly.

5. Possible Overfitting:
   - If cross-validation was used, it should mitigate overfitting to some extent. However, if 'sports' is overrepresented in each fold, the model may still overfit to this category.

6. Need for Data Resampling or Model Adjustment:
   - Strategies like resampling the data to have a balanced distribution or applying class weights in the model might improve the classifier's performance on underrepresented categories.

7. Validity of the Cross-Validation:
   - It would be essential to ensure that cross-validation was implemented correctly. Each fold should represent the category distribution of the whole dataset to give a valid performance estimate.

8. Assessment of Data Quality:
   - It's important to assess the quality and consistency of the labeled data. Mislabeling or inconsistencies can lead to poor model performance.

### LDA
1. Topic Distribution:
   - The LDA model seems to have identified three distinct topics. The words associated with Topic 0 are related to movies, Topic 1's words are associated with books, and Topic 2's words seem to be a mix but likely still related to books. However, the presence of generic terms like 'just', 'good', 'great', and 'like' across multiple topics indicates that some common words might be too prevalent and could be obscuring the differentiation between topics.

2. LDA Model Clarity:
   - The identified topics suggest that while the LDA model can capture some of the theme-specific words, it also includes general terms that are not very informative for topic distinction. This could be improved by further preprocessing steps such as stopword removal or adjusting the model's hyperparameters.

3. Evaluation Metrics Interpretation:
   - The bar chart appears to be showing precision, recall, F1-score, and support for three different classes (presumably book, movie, sports), which suggests a classification task has been performed post-LDA topic modeling.
   - The support bar for the 'sports' category is much higher than for 'book' and 'movie', indicating more samples and potentially skewing the model toward this category.
   - The 'sports' category has a higher F1-score compared to 'book' and 'movie', which might suggest that the LDA model is better at distinguishing sports-related documents, or the classifier trained on LDA's output is performing better for sports.

4. Model Performance and Data Quality:
   - Given the high precision and recall for the 'sports' category, it appears that the classifier is performing well in identifying and classifying sports-related documents. However, the performance on 'book' and 'movie' categories is not as good, which could be due to fewer training samples or less distinctive topic word distributions.
   - The low recall for the 'book' category implies many book-related documents are being misclassified, whereas the low precision for 'movie' suggests that non-movie documents are being incorrectly classified as movies.

5. Cross-validation Effectiveness:
   - If cross-validation was used, it should help ensure that the model's performance metrics are robust across different subsets of the data. However, the imbalance in class representation still needs to be addressed.

From these observations, actions like rebalancing the dataset, enhancing the preprocessing of text data, or refining the LDA model's parameters might be necessary to improve the overall quality and distinctiveness of the topic representation. Additionally, it is essential to verify the classification step's accuracy and consider using stratified folds in cross-validation to maintain the class distribution.

