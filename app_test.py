from altair.vegalite.v4.schema import Month
import streamlit as st
import numpy as np
import pandas as pd
import hydralit as hy
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import seaborn as sns # for Interactive plots
import itertools
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

### Load Dataset #####################

df_general = pd.read_csv('general_data.csv')
df_employee=pd.read_csv('employee_survey_data.csv')
df_manager=pd.read_csv('manager_survey_data.csv')
df_manager['total_mn'] = (df_manager['JobInvolvement']+df_manager['PerformanceRating'])/2
df_employee['total_em'] = (df_employee['EnvironmentSatisfaction']+df_employee['JobSatisfaction']+df_employee['WorkLifeBalance'])/3

df_general = df_general.merge(df_manager[['EmployeeID', 'JobInvolvement', 'PerformanceRating','total_mn']], on='EmployeeID')
df_general = df_general.merge(df_employee[['EmployeeID', 'EnvironmentSatisfaction', 'JobSatisfaction','WorkLifeBalance','total_em']], on='EmployeeID')
df_general.dropna(inplace=True)
df_general.drop(columns=['EmployeeCount','Over18','StandardHours','EmployeeID'], axis = 1, inplace = True)


# create a list of our conditions
conditions = [
    (df_general['Age'] <= 20),
    (df_general['Age'] > 20) & (df_general['Age'] <= 30),
    (df_general['Age'] > 30) & (df_general['Age'] <= 40),
    (df_general['Age'] > 40) & (df_general['Age'] <= 50),
    (df_general['Age'] > 50) & (df_general['Age'] <= 60)
    ]

# create a list of the values we want to assign for each condition
values = ['less_20', '20_30', '30_40', '40_50', '50_60']
df_general['Age_tier'] = np.select(conditions, values)


df_general["Attrition"].replace({"Yes": 1, "No": 0}, inplace=True)
df_general.groupby(['Age_tier']).mean()
df_G_Age = df_general.groupby('Age_tier').mean()[['Attrition','MonthlyIncome','PercentSalaryHike','StockOptionLevel','JobLevel','TrainingTimesLastYear','TotalWorkingYears','YearsSinceLastPromotion','YearsWithCurrManager','PerformanceRating']]

### data cleaning again
df_hr=pd.read_csv('hr_final.csv')
df_hr.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)
others = df_general.select_dtypes('object').columns
le = LabelEncoder()
for col in others:
    df_general[col] = le.fit_transform(df_general[col])



df_analysis = df_hr[['Age','BusinessTravel','Department','Education', 'EducationField', 'Gender', 'JobLevel','MaritalStatus','NumCompaniesWorked','PercentSalaryHike', 'StockOptionLevel','TrainingTimesLastYear','YearsSinceLastPromotion','YearsWithCurrManager', 'JobInvolvement', 'PerformanceRating','EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance',
       'Age_tier', 'distancehome_tier', 'MonthlyIncome_tier','total_mn','total_em', 'Attrition']]

################ ML preprocessing #################################
X = df_hr.drop(['Attrition'] ,axis =1)
y = df_hr[['Attrition']]
print(X.shape, y.shape)

################ scatter chart ####################################
st.markdown("## HR Attrition Analysis") 

st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Features :")


features = df_general.drop(labels=["Attrition"], axis=1).columns.tolist()

x_axis = st.sidebar.selectbox("X-Axis", features)
y_axis = st.sidebar.selectbox("Y-Axis", features, index=1)

if x_axis and y_axis:
    scatter_fig = plt.figure(figsize=(8,7))

    scatter_ax = scatter_fig.add_subplot(111)

    Attrition_with_df = df_general[df_general["Attrition"] == 1]
    Attrition_no_df = df_general[df_general["Attrition"] == 0]

    Attrition_with_df.plot.scatter(x=x_axis, y=y_axis, s=50, c="tomato", alpha=0.6, ax=scatter_ax, label="Attrition=Yes")
    Attrition_no_df.plot.scatter(x=x_axis, y=y_axis, s=50, c="dodgerblue", alpha=0.6, ax=scatter_ax,
                       title="{} vs {}".format(x_axis.capitalize(), y_axis.capitalize()), label="Attrition=No")

################ bar chart ####################################
st.sidebar.markdown("### Bar Chart: Average Features : ")
avg_df_analysis=df_analysis.groupby("Attrition").mean()
bar_axis = st.sidebar.multiselect(label="Average Features With Type Bar Chart", options=features, default=['BusinessTravel','Department','Education', 'EducationField'])
if bar_axis:
    bar_fig = plt.figure(figsize=(5,4))

    bar_ax = bar_fig.add_subplot(111)
    sub_avg_df_analysis = avg_df_analysis[bar_axis]
    sub_avg_df_analysis.plot.bar(alpha=0.8, ax=bar_ax, title="Attrition by per working hour value");

else:
    bar_fig = plt.figure(figsize=(5,4))

    bar_ax = bar_fig.add_subplot(111)

    sub_avg_df_analysis = avg_df_analysis[['BusinessTravel','Department','Education', 'EducationField']]

    sub_avg_df_analysis.plot.bar(alpha=0.8, ax=bar_ax, title="Average Measurements");

#################### hist chart ####################################
st.sidebar.markdown("### Histogram: Distribution of Features : ")
hist_axis = st.sidebar.multiselect(label="Histogram", options=features, default=['total_mn','total_em'])
bins = st.sidebar.radio(label="Bins :", options=[2,4,6,8,10], index=4)
if hist_axis:
    hist_fig = plt.figure(figsize=(6,4))

    hist_ax = hist_fig.add_subplot(111)

    sub_df_analysis = df_analysis[hist_axis]
    sub_df_analysis.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution");

else:
    hist_fig = plt.figure(figsize=(6,4))

    hist_ax = hist_fig.add_subplot(111)

    sub_df_analysis = df_analysis[['total_mn','total_em']]

    sub_df_analysis.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution");


#################### Hexbin chart ####################################
st.sidebar.markdown("### Hexbin Chart: Concentration of Features :")

hexbin_x_axis = st.sidebar.selectbox("Hexbin-X-Axis", features, index=0)
hexbin_y_axis = st.sidebar.selectbox("Hexbin-Y-Axis", features, index=1)

if hexbin_x_axis and hexbin_y_axis:
    hexbin_fig = plt.figure(figsize=(6,4))

    hexbin_ax = hexbin_fig.add_subplot(111)

    df_analysis.plot.hexbin(x=hexbin_x_axis, y=hexbin_y_axis,
                                 reduce_C_function=np.mean,
                                 gridsize=25,
                                 #cmap="Greens",
                                 ax=hexbin_ax, title="Concentration of Features");


##################### Layout Application ##################

container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        scatter_fig
    with col2:
        bar_fig


container2 = st.container()
col3, col4 = st.columns(2)

with container2:
    with col3:
        hist_fig
    with col4:
        hexbin_fig


################ our analysis(under 30 age: I will put some chart)##############################
corr = df_general.corr()
plt.figure(figsize=(25,10))
sns.heatmap(corr,annot=True,cbar=True,cmap="coolwarm")
plt.xticks(rotation=90)
st.title ("Correlation between variables")
fig, ax = plt.subplots(figsize=(11, 9))
corr = df_general.corr()
sns.heatmap(corr,cmap="coolwarm")
st.write(fig)


st.subheader("According to the correlation, we look into **Attrition** and **Age**.")
st.dataframe(df_G_Age.style.highlight_max(axis=0))

st.subheader ("Average Attrition : 16%")



st.write (" But the Attrition of the under 30 was higher than the average.\nThis means that the companys employment loss is large.\nWe found out that")
st.markdown ("- under 30 age")
st.markdown ("- Department: Research & Development, Sales")
st.markdown ("- Job level is higher than average")  
st.markdown ("- Monthly Income less than average (about 20%)")
st.markdown ("- Job involvement is high, but Enviroment and Jobsatisfation are low.")
st.markdown ("Conclusion : People with such characteristics will leave the company.\n And that is a huge loss for the company. So we will have to make sure that does not happen.")

###################################################################################################
#st.sidebar.radio ('Pick your gender',['Male','Female'])
#st.sidebar.select_slider('Choose education', ['highschool', 'Bachelor', 'Master','Over PHD'])
#st.sidebar.number_input('Distance from company', 0,10)
#st.sidebar.text_input('Salary')

####################### USML analysis ##################################################

st.header("HR_Attrition USML Analysis") 
st.markdown("We select **K_means** model")

Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
# plt.plot(K, Sum_of_squared_distances, 'bx-')
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)
fig ,ax = plt.subplot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
st.line_chart(fig)







######################## SML analysis ###################################################
st.header("HR_Attrition SML Analysis") 
st.markdown("We select **Random Forest** Model")