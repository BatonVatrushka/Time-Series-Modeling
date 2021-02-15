import pandas as pd
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns

# use the api to pull in the data from FRED
# api_key=329961d88c12ef9d91df6cc8e64c6900
fred = Fred(api_key='329961d88c12ef9d91df6cc8e64c6900')

# get the data for E-Commerce Retail Sales
e_commerce = fred.get_series('ECOMNSA')

# get the data for e-commerce retails sales
# as % of total sales
e_commerce_per_total_sales = fred.get_series('ECOMPCTNSA')

# get the data for electronic retail sales
electronic_shopping = fred.get_series('MRTSSM4541USN')

# get the data for retail sales: book stores
book_stores = fred.get_series('MRTSSM451211USN')

# plot the e-commerce retail sales
e_commerce.plot(grid=True)
plt.title('E-Commerce Retail Sales')
plt.show()

# plot the electronic shopping data
electronic_shopping.plot(grid=True)
plt.title('Electronic Shopping Retail Sales')
plt.show()

# plot the book sales data
book_stores.plot(grid=True)
plt.title('Retail Sales: Book Stores')
plt.show()
#-------------------------------------------------------------
# % Changes Over Time
#-------------------------------------------------------------
# create a df w/ the % change year over year
df_ecommerce = pd.DataFrame(e_commerce)
df_ecommerce.columns = ['Retail_Sales']
df_ecommerce['%_Change'] = df_ecommerce['Retail_Sales'].pct_change()

# plot the % change for e-commerce
df_ecommerce['%_Change'].plot(grid=True)
plt.title('E-Commerce Retail Sales % Change Year Over Year')
plt.show()

# plot the % change for E-commerce Retails Sales
# as % of Total Sales
e_commerce_per_total_sales.plot(grid=True)
plt.title('E-Commerce as a % of Total Retail Sales')
plt.show()
