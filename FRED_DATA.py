import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns

# use the api to pull in the data from FRED
# api_key=329961d88c12ef9d91df6cc8e64c6900
fred = Fred(api_key='329961d88c12ef9d91df6cc8e64c6900')

# get the data for E-Commerce Retail Sales
e_commerce = fred.get_series('ECOMNSA')

# plot the data
e_commerce.plot(grid=True)
plt.title('E-Commerce Retail Sales')
plt.show()

# create a df w/ the % change year over year
df_ecommerce = pd.DataFrame(e_commerce)
df_ecommerce.columns = ['Retail_Sales']
df_ecommerce['%_Change'] = df_ecommerce['Retail_Sales'].pct_change()

# plot the % change
df_ecommerce['%_Change'].plot(grid=True)
plt.title('E-Commerce Retail Sales % Change Year Over Year')
plt.show()