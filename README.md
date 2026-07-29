# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd
df= pd.read_csv("/content/SAMPLEIDS ().csv")
df
```

<img width="835" height="666" alt="Screenshot 2026-07-29 220839" src="https://github.com/user-attachments/assets/6b75a3db-c35b-40d4-8973-4e2ce61a2cd2" />

```
df.head(3)
```

<img width="827" height="141" alt="Screenshot 2026-07-29 221721" src="https://github.com/user-attachments/assets/493417b6-1f3e-402c-bc35-d8e0f52ed07d" />

```
df.tail(5)
```

<img width="832" height="195" alt="Screenshot 2026-07-29 221812" src="https://github.com/user-attachments/assets/d513b906-76bf-4f5f-acff-5603c3b02062" />

```
df.isnull()
```

<img width="833" height="717" alt="Screenshot 2026-07-29 221900" src="https://github.com/user-attachments/assets/5f37391a-586e-45cc-ab8d-1b31169539cb" />

```
df.notnull()
```

<img width="833" height="747" alt="Screenshot 2026-07-29 222000" src="https://github.com/user-attachments/assets/49be04b7-8e8d-4469-b2a4-4272b810dc81" />

```
df.isnull().sum()
```

<img width="250" height="575" alt="Screenshot 2026-07-29 222111" src="https://github.com/user-attachments/assets/d672809b-76cd-42d9-9e37-9c4065be4f08" />

```
df.isnull().any()
```

<img width="287" height="572" alt="Screenshot 2026-07-29 222220" src="https://github.com/user-attachments/assets/f5d871b0-07b5-4705-a7e2-c86f65b8375a" />

```
df.dropna()
```

<img width="832" height="416" alt="Screenshot 2026-07-29 222309" src="https://github.com/user-attachments/assets/c1954d12-ce5a-4609-98e5-0663ff30194f" />

```
df.dropna(axis=1)
```

<img width="402" height="712" alt="Screenshot 2026-07-29 222358" src="https://github.com/user-attachments/assets/9f72138a-42fb-4cc9-aaa2-9ba50305bfcf" />

```
df.fillna(0)
```

<img width="706" height="535" alt="Screenshot 2026-07-29 222444" src="https://github.com/user-attachments/assets/86304cca-8c29-4db2-aee3-44fbd7ceeaae" />

```
df.fillna (method = 'ffill')
```

<img width="703" height="557" alt="Screenshot 2026-07-29 222555" src="https://github.com/user-attachments/assets/8e7cf27d-e4c3-444d-aa12-8fb76bc09d2c" />

```
df.fillna (method = 'bfill')
```

<img width="695" height="552" alt="Screenshot 2026-07-29 222634" src="https://github.com/user-attachments/assets/92bea0e2-19cb-4c8e-93b0-b2b2b93fb577" />

```
ir=pd.read_csv("/content/iris ().csv")
ir
```

<img width="640" height="432" alt="Screenshot 2026-07-29 222729" src="https://github.com/user-attachments/assets/c078e245-35a5-40c3-b3d9-6f891950479b" />

```
ir.describe()
```

<img width="660" height="310" alt="Screenshot 2026-07-29 222801" src="https://github.com/user-attachments/assets/ef3d112e-c3a6-4faf-b3b9-d995b074906d" />

```
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
```

<img width="671" height="587" alt="Screenshot 2026-07-29 222846" src="https://github.com/user-attachments/assets/07888e9b-85d1-40d6-afb2-52308efe69f6" />

```
Q1= ir.sepal_width.quantile(0.25)
Q3= ir.sepal_width.quantile(0.75)
IQR= Q3-Q1
print(IQR )
```

<img width="353" height="35" alt="Screenshot 2026-07-29 222926" src="https://github.com/user-attachments/assets/0e23d9c1-c408-48c2-989b-05276a2f54d4" />

```
rid=ir[((ir.sepal_width<(Q1-1.5*IQR))|(ir.sepal_width>(Q3+1.5*IQR)))]
rid['sepal_width']
```

<img width="207" height="218" alt="Screenshot 2026-07-29 223003" src="https://github.com/user-attachments/assets/f581ad69-fdc9-4830-b6aa-025e6819e830" />

```
rid=ir[~((ir.sepal_width<(Q1-1.5*IQR))|(ir.sepal_width>(Q3+1.5*IQR)))]
rid['sepal_width']
```

<img width="245" height="473" alt="Screenshot 2026-07-29 223042" src="https://github.com/user-attachments/assets/e75f023b-0397-4962-9849-309463b9c29d" />

```
sns.boxplot(x='sepal_width',data= rid)
```

<img width="655" height="488" alt="Screenshot 2026-07-29 223214" src="https://github.com/user-attachments/assets/2e9cce0a-265e-44f4-8c43-dc3aebc2900a" />

```
import numpy as np
import scipy.stats as stats
Z= np.abs(stats.zscore(ir['sepal_width']))
z
```

<img width="695" height="557" alt="Screenshot 2026-07-29 223240" src="https://github.com/user-attachments/assets/83b64ccc-d47a-404c-b43d-a1f42008a66c" />

```
df1 = ir[Z<3]
df1

```
<img width="650" height="455" alt="Screenshot 2026-07-29 223345" src="https://github.com/user-attachments/assets/2bb1b7d7-e355-4b77-8edb-1ddd68640808" />


 
# Result
The given data has been successfully read, cleaned by handling duplicates and missing values, and saved to a new file named cleaned_data.csv.
