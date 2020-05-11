import pandas as pd
test = pd.DataFrame()
filename = []
type = []
for i in range(6671):
    filename.append(i)
    type.append(0)
test['filename'] = filename
test['label'] = type
test.to_csv("../test.csv",index=False)

