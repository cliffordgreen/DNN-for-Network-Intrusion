
# Proposal for Deep Learning Approach to Network Intrusion Detection:

### Evaluation of Generative Architecture Models vs Discriminative Architecture Models

Basic DNN

By-Cliff Green


```python
# Here are some imports that are used along this notebook
import math
import itertools
import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from collections import OrderedDict
import glob
import tensorflow as tf
```


```python
train20_nsl_kdd_dataset_path = "NSL-KDD-Dataset-master/NSL-KDD-Dataset-master/KDDTrain+_20Percent.txt"
train_nsl_kdd_dataset_path = "NSL-KDD-Dataset-master/NSL-KDD-Dataset-master/KDDTrain+.txt"
test_nsl_kdd_dataset_path = "NSL-KDD-Dataset-master/NSL-KDD-Dataset-master/KDDTest-21.txt"


col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","attrib43"])


attack_dict = {
    'normal': 'normal',
    
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',
    
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',    
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}


# In[3]:


def _label2(x):
    if x['labels'] == 'normal':
        return 'normal'
    else:
        return 'attack'

def returnvalue(x):
    return attack_dict.get(x['labels'])


df_kdd_dataset_train = pd.read_csv(train_nsl_kdd_dataset_path, index_col=None, header=0, names=col_names)
df_kdd_dataset_test = pd.read_csv(test_nsl_kdd_dataset_path, index_col=None, header=0, names=col_names)

df_kdd_dataset_train['label2'] = df_kdd_dataset_train.apply(_label2,axis=1)
df_kdd_dataset_train['label3'] = df_kdd_dataset_train.apply(returnvalue,axis=1)

df_kdd_dataset_test['label2'] = df_kdd_dataset_test.apply(_label2,axis=1)
df_kdd_dataset_test['label3'] = df_kdd_dataset_test.apply(returnvalue,axis=1)
```


```python
#df_kdd_dataset_train = df_kdd_dataset_train.dropna()
```


```python
#df_kdd_dataset_train.isnull().values.any()
```


```python
#df_kdd_dataset_train['protocol_type'].unique()
```


```python
#col_names[binary_inx]
```


```python
nominal_inx = [1, 2, 3]
binary_inx = [6, 11, 13, 14, 20, 21]
numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))
```


```python
numeric_inx = [0,
 4,
 5,
 7,
 8,
 9,
 10,
 12,
 15,
 16,
 17,
 18,
 22,
 23,
 24,
 25,
 26,
 27,
 28,
 29,
 30,
 31,
 32,
 33,
 34,
 35,
 36,
 37,
 38,
 39,
 40]
```


```python
catagorical_inx = [1, 2, 3, 6, 11, 13, 14, 20, 21]
```


```python
cols_to_norm = col_names[numeric_inx]
```


```python
df_kdd_dataset_train[cols_to_norm] = df_kdd_dataset_train[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min() ))
df_kdd_dataset_test[cols_to_norm] = df_kdd_dataset_test[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min() ))

```


```python
import tensorflow as tf
```


```python
Numeric = []
for i in numeric_inx+binary_inx:
    #print(i)
    i = tf.feature_column.numeric_column(col_names[i])
    Numeric.append(i)
```


```python
len(df_kdd_dataset_train[col_names[1]].unique())
```




    3




```python
Nominal = []
for i in nominal_inx: 
    i = tf.feature_column.categorical_column_with_hash_bucket(col_names[i],hash_bucket_size=len(df_kdd_dataset_train[col_names[i]].unique()))
    Nominal.append(i)
```


```python
#assign_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
```


```python
#assign_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)
```


```python
#making buckets for age varaible
#age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])
```


```python
feat_cols = Nominal+Numeric
```


```python
X_train = df_kdd_dataset_train.drop(['labels','attrib43','label2','label3','num_outbound_cmds'], axis = 1)

X_test = df_kdd_dataset_test.drop(['labels','attrib43','label2','label3','num_outbound_cmds'], axis = 1)
```


```python
y_train = df_kdd_dataset_train['label3']

y_test = df_kdd_dataset_test['label3']

```


```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['normal', 'DoS','Probe', 'R2L','U2R'])
y_train = le.transform(y_train)
y_test = le.transform(y_test)

```


```python
y_test = pd.DataFrame(y_test)
y_train = pd.DataFrame(y_train)
```


```python
#from sklearn.model_selection import train_test_split
```


```python
#X_train, X_test, y_train, y_test = train_test_split(x_data, labels,  test_size = .3, random_state = 101)
```


```python
X_train.shape
```




    (17633, 40)




```python
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100, num_epochs=1000, shuffle=True)
```

## Shallow Net


```python
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=6)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\Green\AppData\Local\Temp\tmpnritn_33
    INFO:tensorflow:Using config: {'_tf_random_seed': 1, '_save_summary_steps': 100, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_keep_checkpoint_max': 5, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_model_dir': 'C:\\Users\\Green\\AppData\\Local\\Temp\\tmpnritn_33', '_session_config': None}
    


```python
model.train(input_fn=input_func, steps = 1000)
```

    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Saving checkpoints for 1 into C:\Users\Green\AppData\Local\Temp\tmpnritn_33\model.ckpt.
    INFO:tensorflow:step = 1, loss = 179.176
    INFO:tensorflow:global_step/sec: 172.841
    INFO:tensorflow:step = 101, loss = 13.5142 (0.580 sec)
    INFO:tensorflow:global_step/sec: 244.464
    INFO:tensorflow:step = 201, loss = 21.7895 (0.410 sec)
    INFO:tensorflow:global_step/sec: 212.652
    INFO:tensorflow:step = 301, loss = 8.89953 (0.470 sec)
    INFO:tensorflow:global_step/sec: 185.37
    INFO:tensorflow:step = 401, loss = 5.72815 (0.538 sec)
    INFO:tensorflow:global_step/sec: 246.884
    INFO:tensorflow:step = 501, loss = 6.27231 (0.405 sec)
    INFO:tensorflow:global_step/sec: 232.459
    INFO:tensorflow:step = 601, loss = 8.68018 (0.430 sec)
    INFO:tensorflow:global_step/sec: 245.674
    INFO:tensorflow:step = 701, loss = 3.76325 (0.407 sec)
    INFO:tensorflow:global_step/sec: 239.17
    INFO:tensorflow:step = 801, loss = 4.85086 (0.418 sec)
    INFO:tensorflow:global_step/sec: 235.764
    INFO:tensorflow:step = 901, loss = 5.69181 (0.424 sec)
    INFO:tensorflow:Saving checkpoints for 1000 into C:\Users\Green\AppData\Local\Temp\tmpnritn_33\model.ckpt.
    INFO:tensorflow:Loss for final step: 3.82577.
    




    <tensorflow.python.estimator.canned.linear.LinearClassifier at 0x12ad5e3d0b8>




```python
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
```


```python
results = model.evaluate(eval_input_func)
```

    INFO:tensorflow:Starting evaluation at 2017-09-28-22:15:27
    INFO:tensorflow:Restoring parameters from C:\Users\Green\AppData\Local\Temp\tmpnritn_33\model.ckpt-1000
    INFO:tensorflow:Finished evaluation at 2017-09-28-22:15:31
    INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.524854, average_loss = 2.32314, global_step = 1000, loss = 23.2294
    


```python
results
```




    {'accuracy': 0.52485442,
     'average_loss': 2.323139,
     'global_step': 1000,
     'loss': 23.229429}




```python
pred_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size=10,num_epochs=1, shuffle=False)
```


```python
predictions = model.predict(pred_input_func)
```


```python
#list(predictions)
```

## We must go deeper


```python
#for catagory columns
size = [3,66,11]

emb_Nominal = []
for j in range(len(Nominal)):
    emb_Nominal.append(tf.feature_column.embedding_column(Nominal[j], dimension=size[j]))

emb_Nominal    
#embedded_service = tf.feature_column.embedding_column(service, dimension=66)
#embedded_flag = tf.feature_column.embedding_column(flag, dimension=11)

#embed_nominal = [embedded_pt,embedded_service,embedded_flag ]
```




    [_EmbeddingColumn(categorical_column=_HashedCategoricalColumn(key='protocol_type', hash_bucket_size=3, dtype=tf.string), dimension=3, combiner='mean', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x0000012AC9ADB588>, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True),
     _EmbeddingColumn(categorical_column=_HashedCategoricalColumn(key='service', hash_bucket_size=66, dtype=tf.string), dimension=66, combiner='mean', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x0000012AC9ADB0F0>, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True),
     _EmbeddingColumn(categorical_column=_HashedCategoricalColumn(key='flag', hash_bucket_size=11, dtype=tf.string), dimension=11, combiner='mean', initializer=<tensorflow.python.ops.init_ops.TruncatedNormal object at 0x0000012AC9ADBE48>, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None, trainable=True)]




```python
feat_cols = emb_Nominal + Numeric
```


```python
input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)
```


```python
dnn_model = tf.estimator.DNNClassifier(hidden_units=[40,100,10], feature_columns=feat_cols, n_classes=6)
```

    INFO:tensorflow:Using default config.
    WARNING:tensorflow:Using temporary folder as model directory: C:\Users\Green\AppData\Local\Temp\tmpp14cy7u7
    INFO:tensorflow:Using config: {'_tf_random_seed': 1, '_save_summary_steps': 100, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_keep_checkpoint_max': 5, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_model_dir': 'C:\\Users\\Green\\AppData\\Local\\Temp\\tmpp14cy7u7', '_session_config': None}
    


```python
dnn_model.train(input_fn=input_func, steps=10000)
```

    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Restoring parameters from C:\Users\Green\AppData\Local\Temp\tmpp14cy7u7\model.ckpt-1000
    INFO:tensorflow:Saving checkpoints for 1001 into C:\Users\Green\AppData\Local\Temp\tmpp14cy7u7\model.ckpt.
    INFO:tensorflow:step = 1001, loss = 0.120365
    INFO:tensorflow:global_step/sec: 312.654
    INFO:tensorflow:step = 1101, loss = 0.117332 (0.324 sec)
    INFO:tensorflow:global_step/sec: 317.62
    INFO:tensorflow:step = 1201, loss = 0.163696 (0.315 sec)
    INFO:tensorflow:global_step/sec: 340.39
    INFO:tensorflow:step = 1301, loss = 0.0819727 (0.291 sec)
    INFO:tensorflow:global_step/sec: 308.779
    INFO:tensorflow:step = 1401, loss = 0.18231 (0.326 sec)
    INFO:tensorflow:global_step/sec: 361.39
    INFO:tensorflow:step = 1501, loss = 0.232271 (0.278 sec)
    INFO:tensorflow:global_step/sec: 342.731
    INFO:tensorflow:step = 1601, loss = 0.0479929 (0.291 sec)
    INFO:tensorflow:global_step/sec: 349.904
    INFO:tensorflow:step = 1701, loss = 0.277765 (0.287 sec)
    INFO:tensorflow:global_step/sec: 347.546
    INFO:tensorflow:step = 1801, loss = 0.674632 (0.287 sec)
    INFO:tensorflow:global_step/sec: 349.944
    INFO:tensorflow:step = 1901, loss = 0.261073 (0.286 sec)
    INFO:tensorflow:global_step/sec: 348.721
    INFO:tensorflow:step = 2001, loss = 0.240863 (0.287 sec)
    INFO:tensorflow:global_step/sec: 360.015
    INFO:tensorflow:step = 2101, loss = 0.0961032 (0.278 sec)
    INFO:tensorflow:global_step/sec: 352.453
    INFO:tensorflow:step = 2201, loss = 0.121923 (0.284 sec)
    INFO:tensorflow:global_step/sec: 332.417
    INFO:tensorflow:step = 2301, loss = 0.817194 (0.301 sec)
    INFO:tensorflow:global_step/sec: 356.217
    INFO:tensorflow:step = 2401, loss = 0.33536 (0.281 sec)
    INFO:tensorflow:global_step/sec: 360.063
    INFO:tensorflow:step = 2501, loss = 0.0583098 (0.278 sec)
    INFO:tensorflow:global_step/sec: 356.154
    INFO:tensorflow:step = 2601, loss = 0.476226 (0.283 sec)
    INFO:tensorflow:global_step/sec: 360.091
    INFO:tensorflow:step = 2701, loss = 0.113424 (0.276 sec)
    INFO:tensorflow:global_step/sec: 346.299
    INFO:tensorflow:step = 2801, loss = 0.0218085 (0.290 sec)
    INFO:tensorflow:global_step/sec: 335.805
    INFO:tensorflow:step = 2901, loss = 0.0283752 (0.297 sec)
    INFO:tensorflow:global_step/sec: 352.359
    INFO:tensorflow:step = 3001, loss = 0.0608319 (0.285 sec)
    INFO:tensorflow:global_step/sec: 335.808
    INFO:tensorflow:step = 3101, loss = 0.0159847 (0.298 sec)
    INFO:tensorflow:global_step/sec: 342.729
    INFO:tensorflow:step = 3201, loss = 0.516529 (0.292 sec)
    INFO:tensorflow:global_step/sec: 361.353
    INFO:tensorflow:step = 3301, loss = 0.544137 (0.277 sec)
    INFO:tensorflow:global_step/sec: 365.328
    INFO:tensorflow:step = 3401, loss = 0.726524 (0.274 sec)
    INFO:tensorflow:global_step/sec: 320.656
    INFO:tensorflow:step = 3501, loss = 0.100503 (0.312 sec)
    INFO:tensorflow:global_step/sec: 357.478
    INFO:tensorflow:step = 3601, loss = 0.0510661 (0.280 sec)
    INFO:tensorflow:global_step/sec: 357.467
    INFO:tensorflow:step = 3701, loss = 0.0827406 (0.279 sec)
    INFO:tensorflow:global_step/sec: 336.972
    INFO:tensorflow:step = 3801, loss = 0.0019588 (0.298 sec)
    INFO:tensorflow:global_step/sec: 325.93
    INFO:tensorflow:step = 3901, loss = 0.0219649 (0.306 sec)
    INFO:tensorflow:global_step/sec: 372.098
    INFO:tensorflow:step = 4001, loss = 0.107133 (0.272 sec)
    INFO:tensorflow:global_step/sec: 360.092
    INFO:tensorflow:step = 4101, loss = 0.0338428 (0.275 sec)
    INFO:tensorflow:global_step/sec: 347.468
    INFO:tensorflow:step = 4201, loss = 0.276672 (0.289 sec)
    INFO:tensorflow:global_step/sec: 330.251
    INFO:tensorflow:step = 4301, loss = 0.107857 (0.302 sec)
    INFO:tensorflow:global_step/sec: 293.335
    INFO:tensorflow:step = 4401, loss = 0.0588889 (0.339 sec)
    INFO:tensorflow:global_step/sec: 362.669
    INFO:tensorflow:step = 4501, loss = 5.332 (0.279 sec)
    INFO:tensorflow:global_step/sec: 316.618
    INFO:tensorflow:step = 4601, loss = 0.0169896 (0.315 sec)
    INFO:tensorflow:global_step/sec: 352.415
    INFO:tensorflow:step = 4701, loss = 0.00610225 (0.284 sec)
    INFO:tensorflow:global_step/sec: 363.994
    INFO:tensorflow:step = 4801, loss = 0.231748 (0.276 sec)
    INFO:tensorflow:global_step/sec: 345.134
    INFO:tensorflow:step = 4901, loss = 3.04513 (0.290 sec)
    INFO:tensorflow:global_step/sec: 360.014
    INFO:tensorflow:step = 5001, loss = 0.232956 (0.277 sec)
    INFO:tensorflow:global_step/sec: 365.365
    INFO:tensorflow:step = 5101, loss = 0.00488478 (0.275 sec)
    INFO:tensorflow:global_step/sec: 361.318
    INFO:tensorflow:step = 5201, loss = 0.0210275 (0.277 sec)
    INFO:tensorflow:global_step/sec: 334.71
    INFO:tensorflow:step = 5301, loss = 0.46689 (0.299 sec)
    INFO:tensorflow:global_step/sec: 340.358
    INFO:tensorflow:step = 5401, loss = 0.170739 (0.293 sec)
    INFO:tensorflow:global_step/sec: 357.471
    INFO:tensorflow:step = 5501, loss = 0.439198 (0.280 sec)
    INFO:tensorflow:global_step/sec: 361.356
    INFO:tensorflow:step = 5601, loss = 0.207726 (0.278 sec)
    INFO:tensorflow:global_step/sec: 351.21
    INFO:tensorflow:step = 5701, loss = 0.0499044 (0.284 sec)
    INFO:tensorflow:global_step/sec: 345.069
    INFO:tensorflow:step = 5801, loss = 0.111166 (0.290 sec)
    INFO:tensorflow:global_step/sec: 333.588
    INFO:tensorflow:step = 5901, loss = 0.281853 (0.301 sec)
    INFO:tensorflow:global_step/sec: 356.192
    INFO:tensorflow:step = 6001, loss = 0.16039 (0.281 sec)
    INFO:tensorflow:global_step/sec: 345.069
    INFO:tensorflow:step = 6101, loss = 0.383093 (0.290 sec)
    INFO:tensorflow:global_step/sec: 365.328
    INFO:tensorflow:step = 6201, loss = 0.00758045 (0.273 sec)
    INFO:tensorflow:global_step/sec: 352.417
    INFO:tensorflow:step = 6301, loss = 0.0213172 (0.285 sec)
    INFO:tensorflow:global_step/sec: 345.039
    INFO:tensorflow:step = 6401, loss = 0.0178985 (0.289 sec)
    INFO:tensorflow:global_step/sec: 346.33
    INFO:tensorflow:step = 6501, loss = 0.165319 (0.290 sec)
    INFO:tensorflow:global_step/sec: 332.417
    INFO:tensorflow:step = 6601, loss = 0.0846865 (0.300 sec)
    INFO:tensorflow:global_step/sec: 345.136
    INFO:tensorflow:step = 6701, loss = 0.0212607 (0.290 sec)
    INFO:tensorflow:global_step/sec: 356.159
    INFO:tensorflow:step = 6801, loss = 0.0343393 (0.282 sec)
    INFO:tensorflow:global_step/sec: 364.029
    INFO:tensorflow:step = 6901, loss = 0.167552 (0.274 sec)
    INFO:tensorflow:global_step/sec: 363.957
    INFO:tensorflow:step = 7001, loss = 0.0074592 (0.275 sec)
    INFO:tensorflow:global_step/sec: 338.082
    INFO:tensorflow:step = 7101, loss = 0.343431 (0.297 sec)
    INFO:tensorflow:global_step/sec: 357.47
    INFO:tensorflow:step = 7201, loss = 0.559989 (0.280 sec)
    INFO:tensorflow:global_step/sec: 348.721
    INFO:tensorflow:step = 7301, loss = 0.0490432 (0.287 sec)
    INFO:tensorflow:global_step/sec: 325.928
    INFO:tensorflow:step = 7401, loss = 0.0730294 (0.306 sec)
    INFO:tensorflow:global_step/sec: 343.912
    INFO:tensorflow:step = 7501, loss = 0.0018314 (0.290 sec)
    INFO:tensorflow:global_step/sec: 329.183
    INFO:tensorflow:step = 7601, loss = 0.0124444 (0.306 sec)
    INFO:tensorflow:global_step/sec: 358.719
    INFO:tensorflow:step = 7701, loss = 0.0405676 (0.279 sec)
    INFO:tensorflow:global_step/sec: 362.674
    INFO:tensorflow:step = 7801, loss = 0.00291259 (0.276 sec)
    INFO:tensorflow:global_step/sec: 351.175
    INFO:tensorflow:step = 7901, loss = 0.347051 (0.285 sec)
    INFO:tensorflow:global_step/sec: 360.065
    INFO:tensorflow:step = 8001, loss = 0.227074 (0.277 sec)
    INFO:tensorflow:global_step/sec: 362.657
    INFO:tensorflow:step = 8101, loss = 0.439876 (0.276 sec)
    INFO:tensorflow:global_step/sec: 352.416
    INFO:tensorflow:step = 8201, loss = 1.14407 (0.285 sec)
    INFO:tensorflow:global_step/sec: 361.389
    INFO:tensorflow:step = 8301, loss = 0.337501 (0.276 sec)
    INFO:tensorflow:global_step/sec: 357.434
    INFO:tensorflow:step = 8401, loss = 0.603383 (0.280 sec)
    INFO:tensorflow:global_step/sec: 358.795
    INFO:tensorflow:step = 8501, loss = 0.352331 (0.280 sec)
    INFO:tensorflow:global_step/sec: 361.314
    INFO:tensorflow:step = 8601, loss = 0.206674 (0.277 sec)
    INFO:tensorflow:global_step/sec: 362.669
    INFO:tensorflow:step = 8701, loss = 0.0221276 (0.278 sec)
    INFO:tensorflow:global_step/sec: 366.677
    INFO:tensorflow:step = 8801, loss = 0.0751633 (0.270 sec)
    INFO:tensorflow:global_step/sec: 365.363
    INFO:tensorflow:step = 8901, loss = 0.0319816 (0.275 sec)
    INFO:tensorflow:global_step/sec: 363.958
    INFO:tensorflow:step = 9001, loss = 0.405121 (0.275 sec)
    INFO:tensorflow:global_step/sec: 348.714
    INFO:tensorflow:step = 9101, loss = 0.00384744 (0.289 sec)
    INFO:tensorflow:global_step/sec: 372.15
    INFO:tensorflow:step = 9201, loss = 0.0676948 (0.266 sec)
    INFO:tensorflow:global_step/sec: 331.341
    INFO:tensorflow:step = 9301, loss = 0.00133501 (0.303 sec)
    INFO:tensorflow:global_step/sec: 325.928
    INFO:tensorflow:step = 9401, loss = 0.0560638 (0.306 sec)
    INFO:tensorflow:global_step/sec: 346.298
    INFO:tensorflow:step = 9501, loss = 0.108389 (0.289 sec)
    INFO:tensorflow:global_step/sec: 356.188
    INFO:tensorflow:step = 9601, loss = 0.00830353 (0.282 sec)
    INFO:tensorflow:global_step/sec: 362.714
    INFO:tensorflow:step = 9701, loss = 0.0985334 (0.275 sec)
    INFO:tensorflow:global_step/sec: 362.635
    INFO:tensorflow:step = 9801, loss = 0.832755 (0.277 sec)
    INFO:tensorflow:global_step/sec: 349.942
    INFO:tensorflow:step = 9901, loss = 0.0597359 (0.283 sec)
    INFO:tensorflow:global_step/sec: 346.301
    INFO:tensorflow:step = 10001, loss = 0.0106158 (0.291 sec)
    INFO:tensorflow:global_step/sec: 312.643
    INFO:tensorflow:step = 10101, loss = 3.39755 (0.321 sec)
    INFO:tensorflow:global_step/sec: 360.055
    INFO:tensorflow:step = 10201, loss = 0.00933394 (0.278 sec)
    INFO:tensorflow:global_step/sec: 365.324
    INFO:tensorflow:step = 10301, loss = 0.00650295 (0.273 sec)
    INFO:tensorflow:global_step/sec: 349.946
    INFO:tensorflow:step = 10401, loss = 0.0237198 (0.286 sec)
    INFO:tensorflow:global_step/sec: 348.716
    INFO:tensorflow:step = 10501, loss = 0.762347 (0.288 sec)
    INFO:tensorflow:global_step/sec: 319.665
    INFO:tensorflow:step = 10601, loss = 0.0152563 (0.313 sec)
    INFO:tensorflow:global_step/sec: 354.926
    INFO:tensorflow:step = 10701, loss = 0.102227 (0.281 sec)
    INFO:tensorflow:global_step/sec: 357.463
    INFO:tensorflow:step = 10801, loss = 0.0133081 (0.281 sec)
    INFO:tensorflow:global_step/sec: 370.756
    INFO:tensorflow:step = 10901, loss = 0.0347047 (0.269 sec)
    INFO:tensorflow:Saving checkpoints for 11000 into C:\Users\Green\AppData\Local\Temp\tmpp14cy7u7\model.ckpt.
    INFO:tensorflow:Loss for final step: 0.400717.
    




    <tensorflow.python.estimator.canned.dnn.DNNClassifier at 0x12ad4374898>



# Train Data


```python
eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y=y_train, batch_size=100, num_epochs=1, shuffle=False)
```


```python
dnn_model.evaluate(eval_input_func)
```

    INFO:tensorflow:Starting evaluation at 2017-09-28-22:24:08
    INFO:tensorflow:Restoring parameters from C:\Users\Green\AppData\Local\Temp\tmpp14cy7u7\model.ckpt-11000
    INFO:tensorflow:Finished evaluation at 2017-09-28-22:24:14
    INFO:tensorflow:Saving dict for global step 11000: accuracy = 0.988783, average_loss = 0.0322011, global_step = 11000, loss = 3.21939
    




    {'accuracy': 0.98878324,
     'average_loss': 0.032201067,
     'global_step': 11000,
     'loss': 3.2193909}



# Test data


```python
eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y=y_test, batch_size=100, num_epochs=1, shuffle=False)
```


```python
dnn_model.evaluate(eval_input_func)
```

    INFO:tensorflow:Starting evaluation at 2017-09-28-22:24:15
    INFO:tensorflow:Restoring parameters from C:\Users\Green\AppData\Local\Temp\tmpp14cy7u7\model.ckpt-11000
    INFO:tensorflow:Finished evaluation at 2017-09-28-22:24:16
    INFO:tensorflow:Saving dict for global step 11000: accuracy = 0.556503, average_loss = 2.73515, global_step = 11000, loss = 272.343
    




    {'accuracy': 0.55650264,
     'average_loss': 2.7351542,
     'global_step': 11000,
     'loss': 272.3432}



### bad, lets try other models


```python

```
