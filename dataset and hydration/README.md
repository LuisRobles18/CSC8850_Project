# Dataset Source and Hydration

The datasets were obtained from the following source: https://github.com/kglandt/stance-detection-in-covid-19-tweets/tree/main/dataset

There are four datasets (stance with a specific topic related to health mandates):
- Face maks
- Anthony Fauci
- Stay at home orders
- School closures

Training and validation set were already manually labeled, whereas the noisy set were automatically labeled (also called as silver standard dataset). Two approaches were implemented, in which the first approach the silver standard dataset was exclused from the training set. And in the second approach the silver standard dataset was included for the training set.

## Dataset statistics (without silver standard/noisy dataset)

<img src="dataset_statistics_wo_noise.png" />

## Dataset statistics (with silver standard/noisy dataset)

<img src="dataset_statistics_w_noise.png" />

## Hydration process

Since the datasets provided don't include the content of the tweets, an hydration process is required. Therefore, by using the get_metadata.py script from the SMMT (https://github.com/thepanacealab/SMMT) we can hydrate these tweets. It is important to include a JSON file with the API keys provided by Twitter.

After having both the script and the API keys, we just simply execute the following command:

```console
!python get_metadata.py -i dataset/face_masks_test.csv -o hydrated_tweets/face_masks_test_hydrated -k api_keys.json -c 'Tweet Id' -m "e"
```
Where:
- **i** represents the input filename
- **o** represents the output filename
- **k** represents the filename for the API keys in the JSON format
- **m** represents the mode (it is required to introduce "e" for extended mode in order to get the full content of each tweet)

The previous script will generate a CSV files with the hydrated tweets (including the content of the tweet).

### Example

In this folder you'll find a Colab Notebook (*CSC8850_ML_Project_(Hydrating_tweets).ipynb*) showing the process on how these tweets were loaded and hydrated.
