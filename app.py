import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle 

df = pd.read_csv("./data/train.csv")

MODEL_FILE = "./artefacts/clf_fitted.bin"

st.title("Titanic Survivor Predictor")

navigation = st.sidebar.radio(
    "Please chose one of the following options:",
    ["Home", "Data", "Predictions"]
)

if navigation == "Home":
    st.markdown(
    """ ## Welcome to the Titanic Predictor page.
    """
    )
    st.image("https://images.unsplash.com/photo-1612429085511-2c2ac3aac171?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NXx8dGl0YW5pY3xlbnwwfHwwfHw%3D&auto=format&fit=crop&w=800&q=60", width=400)
    
    st.markdown(
    """ ### This model predicts if a person with specific traits has survived the ship's sink.
    """
    )

if navigation == "Data":
    st.write("Welcome to the section on Exploratory Data Analysis.")
    st.write("Data is taken from [titanic](https://www.kaggle.com/competitions/titanic/data?select=train.csv)")
    if st.checkbox("Click here to see the original dataframe"):
        st.title("Titanic training dataset")
        st.dataframe(df)
    if st.checkbox("Click here to see the correlation between variables"):
        fig,ax = plt.subplots()
        st.title("Heatmap features titanic dataset")
        sns.heatmap(df.corr(), ax = ax)
        st.write(fig)
    if st.checkbox("Click here to see distribution of variables"):
        st.title("Distribution features titanic dataset")
        fig = sns.pairplot(df, hue = "Survived")
        st.pyplot(fig)
    if st.checkbox("Click here to see the variables taken into account for model fit"):
        st.title("Features for model fit")
        st.write(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']].columns)

if navigation == "Predictions":
    st.markdown(
    """ #### Welcome to the predictions page.
    """
    )
   
    st.markdown(
    """ ##### For predictions regarding if a passenger survived or not, please answer the following questions. 
    """
    )

    @st.cache
    def load_model():
        with open(MODEL_FILE, "rb") as model_in:
            classifier_logistic_regression = pickle.load(model_in)
        return classifier_logistic_regression

    classifier_logistic_regression = load_model()

    # Using "widgets" for user interaction:

    col1, col2, col3 = st.columns(3)

    gender = col1.number_input(
        "Gender(type '1' for feminine and '0' for masculine)",
        min_value = 0,
        max_value = 1
    )

    age = col2.number_input(
        "Age", 
        min_value = 1,
        max_value = 100,
        )
    siblings = col3.slider(
        "How many siblings does the passenger have? (Choose between '0' and '8')", 
        min_value = 0,
        max_value = 8,
        step = 1
        )

    col4, col5, col6 = st.columns(3)    # if you an define an additional 3 columns, they will stacked below under the first ones

    parch = col4.number_input(
        "How many family member (parents/children) were aboard?", 
        min_value = 1,
        max_value = 10,
        )
    female_pclass = col5.slider(
        "If female: which class did the passenger travel in?", 
        min_value = 0,
        max_value = 3,
        step = 1
        )
    male_pclass = col6.slider(  # !!ANOTHER SLIDER OBJECT!!
        "If male: which class did the passenger travel in?", 
        min_value = 0,
        max_value = 3,
        step = 1,
        key=101     # !! NOTE !! – If you define multiple slider objects, make sure to give it seperate keys. Otherwise it will not be able to seperate them.
        )

    user_input = {
            "gender": int(gender),
            "age": int(age),
            "siblings": int(siblings),
            "parch": int(parch),
            "female_pclass": int(female_pclass),
            "male_pclass": int(male_pclass),
            }

    user_input = pd.DataFrame(
        user_input, 
        columns = user_input.keys(), 
        index = [0]
        )

    user_input = np.array(user_input).reshape(1, -1)

    # pred = classifier_logistic_regression.predict(user_input)[0]
    proba = classifier_logistic_regression.predict_proba(user_input)[0]

    if st.button("Predict"):
        st.success(f'This person would have survived with a probability of: {round(proba[1], 2)}')
        