import pandas as pd


headers = ['Label', 'Time', 'Magnitude', 'Before', 'After', 'Duration', 'Error', 'Trend']
dictionary = dict.fromkeys(headers)
df = pd.DataFrame([dictionary])

df.to_csv(r'D:\Anna\Documents\Uni\Thesis\Code\Detections\Q0000012a_results.csv', index=False, header = True)  

print(df)
