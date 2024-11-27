import  tensorflow as tf
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.preprocessing  import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
#Load the trained model
model=tf.keras.models.load_model('regression_model.h5')
#Load scaler,one hot encode ,label encode pickle files
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_encoding_geo.pkl','rb') as file:
    one_hot_encoding_geo=pickle.load(file)    
    
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    
    
## Streamlit app
st.title("Customer Salary Prediction")
geography=st.selectbox('Geography',one_hot_encoding_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_number=st.selectbox('Is Active Number',[0,1])
is_exited=st.selectbox("Is_Exited",[0,1])

#input data

input_data=pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography'	: [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_number],
    'Exited':[is_exited]
        })
print(input_data)
#geo encoding
geo_input_df = pd.DataFrame([[geography]], columns=['Geography'])
geo_encoded=one_hot_encoding_geo.transform(geo_input_df).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoding_geo.get_feature_names_out(['Geography']))
#concat
input_data_df=pd.concat([input_data.drop(['Geography'],axis=1),geo_encoded_df],axis=1)
#scaling
input_scaled=scaler.transform(input_data_df)
#prediction
# predict churn

prediction=model.predict(input_scaled)
prediction_salary=prediction[0][0]
print(prediction_salary)

st.write("Predicetd Salary:",prediction_salary)