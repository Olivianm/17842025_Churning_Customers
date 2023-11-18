
# Importing libraries
import pandas as pd
import pickle
import streamlit as st

# Load the model
pickle_in = open('Customer_churning.pkl', 'rb')
m_load = pickle.load(pickle_in)

# Define the prediction function
def prediction(features):
    # Convert features to a DataFrame
    df = pd.DataFrame([features])
    
    # Predict churn
    prediction_result = m_load.predict(df)
    
    return prediction_result

# Main function
def main():
    st.title('Customer Churn Prediction Model')
    
    "Features to predict Customer churn prediction"

    # Collect user input
    feature1 = st.selectbox('gender:', ['Female', 'Male'])
    feature2 = st.selectbox('SeniorCitizen:', ['Yes', 'No'])
    feature3 = st.selectbox('Partner:', ['Yes', 'No'])
    feature4 = st.selectbox('Dependents:', ['Yes', 'No'])
    feature5 = st.number_input('tenure:', min_value=0.0, max_value=1000000.0, step=1.0)
    feature6 = st.selectbox('PhoneService:', ['Yes', 'No'])
    feature7 = st.selectbox('MultipleLines:', ['Yes', 'No', 'No Service'])
    feature8 = st.selectbox('InternetService:', ['DSL', 'Fiber Optic', 'No Service'])
    feature9 = st.selectbox('OnlineSecurity:', ['Yes', 'No', 'No internet Service'])
    feature10 = st.selectbox('OnlineBackup:', ['Yes', 'No', 'No internet Service'])
    feature11 = st.selectbox('DeviceProtection:', ['Yes', 'No', 'No internet Service'])
    feature12 = st.selectbox('TechSupport:', ['Yes', 'No', 'No internet Service'])
    feature13 = st.selectbox('StreamingTV:', ['Yes', 'No'])
    feature14 = st.selectbox('StreamingMovies:', ['Yes', 'No'])
    feature15 = st.selectbox('Contract:', ['Month to month', 'One year', 'Two years'])
    feature16 = st.selectbox('PaperlessBilling:', ['Yes', 'No'])
    feature17 = st.selectbox('PaymentMethod:', ['Electronic check', 'Mailed check', 'Bank Transfer', 'Credic card'])
    feature18 = st.number_input('MonthlyCharges:', min_value=0.0, max_value=1000000.0, step=0.001)
    feature19 = st.number_input('TotalCharges:', min_value=0.0, max_value=1000000.0, step=0.001)

    # Predict and display result on button press
    button_predict = st.button("Predict")
    button_reset = st.button("Reset")

    if button_predict:
        # Prepare the features dictionary
        input_features = {
            'gender': feature1,
            'SeniorCitizen': feature2,
            'Partner': feature3,
            'Dependents': feature4,
            'tenure': feature5,
            'PhoneService': feature6,
            'MultipleLines': feature7,
            'InternetService': feature8,
            'OnlineSecurity': feature9,
            'OnlineBackup': feature10,
            'DeviceProtection': feature11,
            'TechSupport': feature12,
            'StreamingTV': feature13,
            'StreamingMovies': feature14,
            'Contract': feature15,
            'PaperlessBilling': feature16,
            'PaymentMethod': feature17,
            'MonthlyCharges': feature18,
            'TotalCharges': feature19
        }

        try:
            # Predict and display the result
            result = prediction(input_features)
            st.write(f"The predicted likelihood of churn is: {result}%")
        except ValueError:
            st.error('Please enter valid numeric values for numeric features.')

if __name__ == "__main__":
    main()




# #Importing libraries

# #from sklearn.neural_network import MLPClassifier
# import pandas as pd
# import numpy as np
# import pickle
# import streamlit as st
# from PIL import Image


# # loading the model 
# pickle_in = open('Customer_churning.pkl', 'rb')
# m_load = pickle.load(pickle_in)


# # defining the function that make the prediction
# def prediction(feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
#                feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19):   
   
#     prediction = m_load.predict( 
#         [[feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
#                feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19]]) 
#     print(prediction) 
#     #return prediction 
    
    

# # main function in which we define our webpage
# def main():
#     st.title('Customer Churn Prediction Model')
    
#     "Features to predict Customer churn prediction"

#     # Collect user input
#     feature1 = st.selectbox('gender:',['Female', 'Male'])
#     feature2 = st.selectbox('SeniorCitizen:', ['Yes' , 'No'])
#     feature3 = st.selectbox('Partner:',['Yes' , 'No'])
#     feature4 = st.selectbox('Dependents:',['Yes' , 'No'])
#     feature5 = st.number_input('tenure:', min_value = 0.00, value=0.00, max_value=1000000.00, step= 0.001)
#     feature6 = st.selectbox('PhoneService:',['Yes' , 'No'])
#     feature7 = st.selectbox('MultipleLines:',['Yes' , 'No', 'No Service'])
#     feature8 = st.selectbox('InternetService:',['DSL', 'Fiber Optic', 'No Service'])
#     feature9 = st.selectbox('OnlineSecurity:',['Yes' , 'No', 'No internet Service'])
#     feature10 = st.selectbox('OnlineBackup:',['Yes' , 'No', 'No internet Service'])
#     feature11 = st.selectbox('DeviceProtection:',['Yes' , 'No', 'No internet Service'])
#     feature12 = st.selectbox('TechSupport:',['Yes' , 'No', 'No internet Service'])
#     feature13 = st.selectbox('StreamingTV:',['Yes' , 'No'])
#     feature14 = st.selectbox('StreamingMovies:',['Yes' , 'No'])
#     feature15 = st.selectbox('Contract:',['Month to month' , 'One year', 'Two years'])
#     feature16 = st.selectbox('PaperlessBilling:',['Yes' , 'No'])
#     feature17 = st.selectbox('PaymentMethod:', ['Electronic check', 'Mailed check', ['Bank Transfer'],
#                                                  'Credic card'])
#     feature18 = st.number_input('MonthlyCharges:',min_value = 0.00, value=0.00 ,max_value=1000000.00, step= 0.001)
#     feature19 = st.number_input('TotalCharges:',min_value = 0.00, value=0.00, max_value=1000000.00, step= 0.001)
    
    
#     button =st.button("Predict")
#     button = st.button("reset")

#     if button:
#         input ={
#             feature1:'gender',
#             feature2: 'SeniorCitizen:',
#             feature3: 'Partner:',
#             feature4:'Dependents:',
#             feature5: 'tenure:', 
#             feature6: 'PhoneService:',
#             feature7: 'MultipleLines:',
#             feature8: 'InternetService:',
#             feature9:'OnlineSecurity:',
#             feature10: 'OnlineBackup:',
#             feature11: 'DeviceProtection:',
#             feature12: 'TechSupport:',
#             feature13: 'StreamingTV:',
#             feature14: 'StreamingMovies:',
#             feature15: 'Contract:',
#             feature16: 'PaperlessBilling:',
#             feature17: 'PaymentMethod:', 
#             feature18: 'MonthlyCharges:',
#             feature19: 'TotalCharges:'
            
#         }
#         df =pd.DataFrame([input])    
#         prediction = m_load.predict( df)
#         print(prediction)
        
#         st.write(f"The predicted likelihood of churn is: {prediction}%")
    
    



# if __name__ == "__main__":
#     main()









   

