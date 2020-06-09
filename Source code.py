import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score as rsq


file1 = r'D:\Work\Projects\COVID-19\covid_19_data.csv'
df = pd.read_csv(file1)

df1 = df.loc[df['Country/Region'] == 'India', ['ObservationDate', 'Confirmed', 'Deaths']]
print ("\nFirst 5 rows:\n",df1.head())
print ("\nLast 5 rows:\n",df1.tail())

new_confirmed = df1["Confirmed"].astype(float)
new_deaths = df1["Deaths"].astype(float)

df1['Date-parsed'] = pd.to_datetime(df1['ObservationDate'],format = "%d%m%y",
   infer_datetime_format=True)
week_of_the_year = df1['Date-parsed'].dt.week
new_week =week_of_the_year.astype(float)


bins = np.linspace(min(week_of_the_year), max(week_of_the_year), 4)
group = ["<week 4","week 5-8","week 8>"]
df1['week-binned'] = pd.cut(week_of_the_year, bins,
                           labels=group, include_lowest = True)
v_counts = df1["week-binned"].value_counts()
v_counts.index.name = "Outbreak Range"


bins = np.linspace(min(df['Confirmed']), max(new_confirmed), 4)
group = ['Mild',"Severe","Critical"]

df1['Confirmed-binned'] = pd.cut(new_confirmed, bins,
                           labels=group, include_lowest = True)
v_counts = df1["Confirmed-binned"].value_counts()
v_counts.index.name = "Confirmed cases outbreak Range"
print("\n", v_counts)

#twinx method

fig,ax = plt.subplots()
ax.plot(df1["Confirmed"],week_of_the_year , color="red")
ax.set_xlabel("Confirmed cases",fontsize=14)
ax.set_ylabel("Weeks ",color="red",fontsize=14)
#plt.xlim(0,12)
#plt.ylim(0,15)
# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
ax2.plot(df1["Confirmed"],df1["Deaths"] ,color="blue",marker="o")
ax2.set_ylabel("Deaths",color="blue",fontsize=14)
#plt.xlim(0,12)
#plt.ylim(0,15)
plt.title("SARS CoV-2 outbreak in India")
plt.show()

#SimpleLinearRegression
X = df1[["Confirmed"]]
Y = week_of_the_year
Z = df1[['Confirmed','Deaths']]
lm = LinearRegression()

lm.fit(X,Y)
#   wlm.fit(Z, week_of_the_year)
Yhat = lm.predict(X)
print("\nIntercept:\n",lm.intercept_)
print("\nCoefficient:\n",lm.coef_)
print("\nThis shows the rise in Confirmed cases has very less impact on average change in weeks.")
print("This implies cases are escalating in shorter period of time.\n")


print("\nRegression Plot:")
sns.regplot(X, Y, data=df1, color="blue", label="A")
sns.regplot(X, Yhat, data=df1, color="red", label="B")
plt.legend(labels=['Actual values', 'Fitted values'])
plt.xlabel('Confirmed cases')
plt.ylabel('Weeks')
plt.title("COVID-19 outbreak in India")
plt.show()

MSE = mse(X, Yhat)
print('\nMSE:\n',MSE)

R_sq = rsq(X, Yhat)
print('\nR_squared:\n',R_sq)
#train_test

x_train, x_test, y_train, y_test = tts(X, Y, test_size=0.2,
                                       random_state=0)
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.transform(x_test)

clf = svm.SVC()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)

print(classification_report(y_test, pred))

print("Accuracy Score:",accuracy_score(y_test, pred))



