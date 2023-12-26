import pandas as pd
['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
file_path = r'C:\Users\14833\Desktop\pytorch-classifier-master\pytorch_classifier_master\Datasets\label_3064.txt'
# df_raw = pd.read_csv(file_path,sep=' ',names=['path','crosssroad','elevate','auxiliary'])
df_raw = pd.read_csv(file_path,sep=' ',names=['path','glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor'])
df_test = df_raw.sample(459)
df_train = df_raw
df_train = df_train.append(df_test)
# df2.append(df1)
# df2.append(df1)
# df_train = df_train.drop_duplicates(subset=['path','crosssroad','elevate','auxiliary'],keep=False)
df_train = df_train.drop_duplicates(subset=['path','glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor'],keep=False)
print(df_raw.shape)
print(df_test.shape)
print(df_train.shape)
df_test.to_csv('./Datasets/label_3064_brain_val.csv',sep=' ',index=None,header=None)
df_train.to_csv('./Datasets/label_3064_brian_train.csv',sep=' ',index=None,header=None)