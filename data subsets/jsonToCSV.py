import pandas as pd

df1 = pd.read_json("cars.json")
df2 = pd.read_json("chairs.json")
df3 = pd.read_json("couch.json")

df = pd.concat([df1, df2, df3])

df.to_csv("subsets.csv", index=False)
