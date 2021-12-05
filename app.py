import streamlit as st
import requests
import pandas as pd

# http://127.0.01:5000/ is from the flask api
response = requests.get("http://0.0.0.0:4000/")
print(response.json())
data_table1 = pd.DataFrame(response.json())
st.write(data_table1)
