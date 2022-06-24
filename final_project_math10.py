
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from PIL import Image

#image = Image.open(r"C:\Users\AViey\OneDrive\Documents\College\UCI\Math 10 intro to data analysis\FinalProject\Customer-Analysis-1.jpg")
#st.image(image, caption='Is what we earn what we consume?')

title = st.title("Predicting Customer Income")
st.markdown("[Alfonso Vieyra's Github](https://github.com/anvieyra/FinalProjectMath10)")
if st.button('Contents'):
    st.write('Purpose - goal of analysis\n')
    st.write('Chosen Dataset\n')
    st.write('Cleaning the Data')
    st.write('Deploying Machine Learning Model')
    st.write('Conclusion')
    
if st.button('Purpose'):
#st.header('Purpose')
    st.write('Hello Everyone!\n\n In this program, I will be using a linear'
             ' regression model to help try and predict what the income of'
        ' a customer might have be based on a set of grocery purchases')
 
if st.button('Our Dataset'):
    st.write('Below is the dataset we will be using. If you click on the link it'
        ' will direct you to the associated dataset on kaggle. We will use this'
        ' for our linear regression model')
    image1 = Image.open(r"C:\Users\AViey\OneDrive\Documents\College\UCI\Math 10 intro to data analysis\FinalProject\MarketCampaignPic.PNG")
    st.markdown('[Kaggle Market Campaign Dataset](https://www.kaggle.com/imakash3011/customer-personality-analysis)')
    st.image(image1, caption= 'Source')
    st.write("There are numerous columns we will remove prior to assessing bad values etc.")
    
#*****************************************************************************
#removing unnecessary columns
df = pd.read_csv(r"C:\Users\AViey\OneDrive\Documents\College\UCI\Math 10 intro to data analysis\FinalProject\marketing_campaign1.txt",
                 sep='\t')
df = df.iloc[:,3:20]
df2 = df.drop(columns=['Kidhome', 'Teenhome', 'Dt_Customer', 'Marital_Status', 'NumDealsPurchases', 
                       'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Recency'])

st.subheader('Our DataFrame')
df_rows = st.slider('Select how many rows:', min_value=0, max_value=len(df2.index))
st.dataframe(df2.iloc[:df_rows,:])
st.write('Here we show a sample of the data before any processing is committed.'
         ' If you look closely at some of the cells they will have NA values')

#remove NaN values
if st.button('Cleaning/structuring the data'):
    #remove NaN values
    print(df2.isna().any(axis=0))
    df2 = df2.replace(np.nan, 0)
    st.write('We change the NA cells to 0. Below we have a histogram drawn to' 
             ' highlight discrepancies')
    hist = alt.Chart(df2).mark_bar().encode(
        alt.X('MntMeatProducts:Q', bin=False),
        y='Income',
        )
    st.altair_chart(hist, use_container_width=True)
    st.write('Do you notice how we have all of these outliers that are very much'
             ' indifferent to the norm of the data? Since we want a fair',
             ' chance of an accurate model it would be best to remove these values'
             ' since they boast high leverage and influence in our data.')
else:
        df2 = df2.replace(np.nan, 0)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df2)
#remove outliers that have high leverage and influence
from scipy import stats
df2 = df2[(np.abs(stats.zscore(df2)) < 3).all(axis=1)]

#*****************************************************************************
#Deploying the model
if st.button('Deploy Machine Learning model'):
    import time
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    st.success('Successful Deployment!!')
    st.balloons()
        

reg = LinearRegression()
X = df2.iloc[:,1:]
y_true = df2['Income'].values
reg.fit(X,y_true)
y_predict = reg.predict(X)
y_predict.shape
df2["y_predict"] = y_predict.copy()


st.subheader('Now, pick your columns and rows and we can see how the data is spread')
xAxis = st.selectbox('Select which numeric column for x: ', df2.columns)
df2_rows = st.slider('Select how many rows:', min_value=0, max_value=len(df2.index))

#*****************************************************************************
#visualization
if st.button("Data Visualization: True Response Plot(Income)"):

    brush = alt.selection_interval()
    chart_data = alt.Chart(df2.iloc[:df2_rows,:]).mark_circle().encode(
        x = xAxis,
        y = "Income",
        color=alt.condition(brush, 'MntWines:N', alt.value('lightgray'))
    ).add_selection(
        brush
    )
    chart_data
    
    st.write('The above chart shows the respective covariate plotted with the true'
             ' response value(Income). Below, however, we super impose '
             'a line chart to show how'
             ' the predicted response fits with the data')
    
    chart_data2 = alt.Chart(df2.iloc[:df2_rows,:]).mark_line().encode(
        x = xAxis,
        y = "y_predict",
        color = alt.value("black"),
    )
    chart_data + chart_data2
#*****************************************************************************
#Conclusion
if st.button('Conclusion'):
    st.write('When we look at the super imposed graphs on display. We notice '
             'that our model and the fitted line is very jagged. This is because what'
             ' we see here is the model overfitting its estimate to match the '
             'data as closely as possible. It shows that it has incredibly high variance'
             ' and very little bias in the model.\nTherefore, it would be very difficult'
             ' to apply this model to new datasets because of its overfitting'
             ' of the data. Thus, unfortunately, we need a better attempt or '
             'approach next time!\n Thank you for visiting!\n\n Kind Regards,\n\n'
             'Alfonso Vieyra')

    
st.sidebar.subheader('What is linear Regression?')
st.sidebar.text('Linear regression is a linear model, e.g. a model that assumes\n '
                'a linear relationship between the input variables (x) and the\n '
                'single output variable (y). More specifically, that y can be\n '
                'calculated from a linear combination of the input variables (x).')

st.sidebar.markdown('[MachineLearningMastery](https://machinelearningmastery.com/linear-regression-for-machine-learning/)')