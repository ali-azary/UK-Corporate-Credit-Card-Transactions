import pandas as pd

import os
folder='merchants'
if not os.path.exists(folder):
    os.makedirs(folder)

# Read the CSV file into a pandas dataframe
df = pd.read_csv('1718Pcard.csv')

# Check for missing values
# print(df.isna().sum())

# cleaning the data
df.dropna(inplace=True)
df['FIN.TRANSACTION DATE'] = pd.to_datetime(df['FIN.TRANSACTION DATE'])
df['FIN.POSTING DATE'] = pd.to_datetime(df['FIN.POSTING DATE'])
df['FIN.TRANSACTION AMOUNT'] = df['FIN.TRANSACTION AMOUNT'].str.replace(',', '').astype(float)
df['FIN.ORIGINAL CURRENCY AMOUNT'] = df['FIN.ORIGINAL CURRENCY AMOUNT'].str.replace(',', '').astype(float)

# plot
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(df['FIN.TRANSACTION DATE'], df['FIN.TRANSACTION AMOUNT'])
plt.savefig('scatter.jpg')
plt.close()

# export 
df.to_csv('cleaned_data.csv', index=False)

# scatter plot of top 10selling merchants
# group data by merchant name and sum transaction amount
grouped = df.groupby('MCH.MERCHANT NAME')['FIN.TRANSACTION AMOUNT'].sum()

# sort merchants by transaction amount and select top 10
top_merchants = grouped.sort_values(ascending=False)[:10].index


# Loop over each merchant and create a scatter plot of transaction date vs amount
for merchant in top_merchants:
    merchant_data = df[df["MCH.MERCHANT NAME"] == merchant]
    plt.figure()
    plt.scatter(merchant_data["FIN.TRANSACTION DATE"], merchant_data["FIN.TRANSACTION AMOUNT"])
    plt.title(merchant)
    plt.xlabel("Transaction Date")
    plt.ylabel("Transaction Amount")
    plt.xticks( rotation=45)
    plt.savefig(folder+'/'+merchant+'-scatter.jpg')

# scatter plot of top 10selling merchants


# filter data to only include transactions from top 10 merchants
filtered = df[df['MCH.MERCHANT NAME'].isin(top_merchants)]

# plot scatter plot of transaction date vs amount for each merchant
plt.close("all")
fig, ax = plt.subplots()
for merchant, group in filtered.groupby('MCH.MERCHANT NAME'):
    ax.scatter(group['FIN.TRANSACTION DATE'], group['FIN.TRANSACTION AMOUNT'], label=merchant)
    
# add legend and axis labels
ax.legend()
ax.set_xlabel('Transaction Date')
ax.set_ylabel('Transaction Amount')

plt.savefig('top10scatter.jpg')
plt.close()

# daily sum and avg sales of top 10merchants
# group filtered data by merchant name and transaction date, and sum transaction amounts
grouped = filtered.groupby(['MCH.MERCHANT NAME', 'FIN.TRANSACTION DATE'])['FIN.TRANSACTION AMOUNT'].sum()



# plot sum of transactions with time for each of these top merchants
for merchant in top_merchants:
    merchant_data = df[df['MCH.MERCHANT NAME'] == merchant]
    merchant_data = merchant_data.groupby(['FIN.TRANSACTION DATE'])['FIN.TRANSACTION AMOUNT'].sum()
    merchant_data.plot(kind='bar', title=merchant, figsize=(10, 5))
    n=min(len(merchant_data),20)
    xticks = range(0, len(merchant_data), int(len(merchant_data)/n))
    xtick_labels = [d.strftime('%Y-%m-%d') for d in merchant_data.index[xticks]]
    plt.xticks(xticks, xtick_labels, rotation=45)
    plt.savefig(folder+'/'+merchant+'-sum-barchart.jpg')
    plt.close()
    
# find the merchants with the most number of trasactions
# Calculate total transactions per merchant
merchant_transactions = df.groupby("MCH.MERCHANT NAME")["FIN.TRANSACTION AMOUNT"].count()
# Calculate total transaction amount per merchant
merchant_totals = df.groupby("MCH.MERCHANT NAME")["FIN.TRANSACTION AMOUNT"].sum()
# Get top 10 merchants with the most transactions
top_merchants = merchant_transactions.nlargest(10)

# Loop through top 10 merchants and print their name and transaction count within date range
for merchant_name in top_merchants.index:
    merchant_df = df[df["MCH.MERCHANT NAME"] == merchant_name]
    num_transactions = top_merchants[merchant_name]
    date_range = f"{merchant_df['FIN.TRANSACTION DATE'].min().date()} to {merchant_df['FIN.TRANSACTION DATE'].max().date()}"
    total_transaction_amount = merchant_totals[merchant_name]
    print(f"{merchant_name}: {num_transactions} transactions ({date_range}), Total Transaction Amount: {total_transaction_amount}")
    
# Plot a bar chart of the top 10 merchants by transaction count and total transaction amount
plt.figure(figsize=(12,8))

top_merchants.plot(kind="bar", rot=45)
plt.title("Top 10 Merchants by Transaction Count")
plt.ylabel("Transaction Count")
plt.xlabel("Merchant Name")

# Add labels for the total transaction amount at the end of each bar
for i, count in enumerate(top_merchants):
    merchant_name = top_merchants.index[i]
    total_transaction_amount = merchant_totals[merchant_name]
    plt.text(i, count + 10, f"£{total_transaction_amount:.2f}", ha="center", fontsize=8)
plt.tight_layout()

plt.savefig('top10transactions.jpg')
plt.close()


# regression and forecast for amazon
from sklearn.linear_model import LinearRegression

# Filter the data for AMAZON UK MARKETPLACE
merchant_df = df[df["MCH.MERCHANT NAME"] == "AMAZON UK MARKETPLACE"]

# Extract the relevant columns
X = merchant_df[["FIN.TRANSACTION DATE"]]
y = merchant_df["FIN.TRANSACTION AMOUNT"]

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make a prediction for the next transaction count and total amount
next_transaction_date = pd.Timestamp("2019-01-01")

import datetime as dt

# convert next_transaction_date to numerical value
epoch = dt.datetime.utcfromtimestamp(0)
next_transaction_date_seconds = (next_transaction_date - epoch).total_seconds()

# predict the number of transactions and total amount for the next month
next_transaction_count = model.predict([[next_transaction_date_seconds]])[0]
next_total_amount = next_transaction_count * merchant_df["FIN.TRANSACTION AMOUNT"].mean()

# Print the predicted values
print(f"Next transaction count: {next_transaction_count}")
print(f"Next total amount: {next_total_amount}")
