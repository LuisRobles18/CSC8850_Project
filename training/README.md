# Training

After pre-processing the tweets, we execute the *training.py* script as follows:

For this example in particular, we are training on ten different models by using the face maks dataset (without noise).

```console
!python training.py 5 0 17 1 df_face_masks_train_preprocessed_wo_noise.tsv "face_maks_wo_noise"
```

Where:

- The 1st parameter represents the total number of folds for the k-fold cross validation
- The 2nd parameter represents the GPU to be used (If there's only one available, by default is the first one, or zero)
- The third and fourth parameter represents the random seed and the random state.
- The fifth parameter represents the file name of the pre-processed dataset.
- The sixth parameter represents the name of the experiment.

The models used for the training process were:
- BERT [1]
- BERT Large [1]
- COVID Twitter BERT [2]
- RoBERTa [3]
- RoBERTa Large [3]
- Support Vector Machines
- Decision Tree
- Logistic Regression
- Naive Bayes
- Random Forest

Moreover, this script also generates their respective reports inside the metrics/result folder, in which we can found (for each model):
- Confusion matrix
- Prediction list
- Classification report

##REFERENCES

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

[2]Müller, M., Salathé, M., and Kummervold, P. E., “COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter”, arXiv e-prints, 2020.

[3] Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. “RoBERTa: A Robustly Optimized BERT Pretraining Approach.” arXiv [cs.CL]. arXiv. http://arxiv.org/abs/1907.11692.
