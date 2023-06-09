# RTlogD

# Setup and dependencies

Dependencies：

- rdkit 2020.09.1.0
- python >=3.6
- pytorch 1.9.0
- dgl  0.8.1
- numpy
- pandas
- tqdm
- dgllife 0.2.9

# Using

# Training

```
python train.py
```

Run the above script to train the data based on DB29-data. The final_model folder gives the trained model of RTlogD. The RT_pre-trained model folder provides the RT model.

# Test

#### **Reproducing results in the paper**

```python
python test_T_data.py 
```

Run the above script to get the results of RTlogD on T-data.

The folder of Ablation_studies gives the model of ablation studies and different training studies, all the predicted results can be found in T-data_predictions(chembl32_logD).csv. The lipo_result provides the prediction results of RTlogD on lipo dataset.

#### Predicting logD values of new data

1. Put the data into example.csv

2. ```
    python test.py 
   ```

3.  The predicted results will be shown in the results folder.

# Author

s20-wangyitian@simm.ac.cn
