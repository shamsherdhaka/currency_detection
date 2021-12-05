import streamlit as st
import request
import pandas as pd

# http://127.0.01:5000/ is from the flask api
response = request.get("http://127.0.01:4000/")
print(response.json())
data_table1 = pd.DataFrame(response.json())
st.write(data_table1)
