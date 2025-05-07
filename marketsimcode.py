""""""  		  	   		 	 	 			  		 			     			  	 
"""MC2-P1: Market simulator.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import datetime as dt  		  	   		 	 	 			  		 			     			  	 
import os  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import numpy as np  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
import pandas as pd  		  	   		 	 	 			  		 			     			  	 
from util import get_data, plot_data  	

def author():
    return "mwang709"

def study_group():
    return "mwang709" 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
def compute_portvals(  		
    symbol,  	   		 	 	 			  		 			     			  	 
    orders_file="./orders/orders.csv",  		  	   		 	 	 			  		 			     			  	 
    start_val=1000000,  		  	   		 	 	 			  		 			     			  	 
    commission=9.95,  		  	   		 	 	 			  		 			     			  	 
    impact=0.005,  		  	   		 	 	 			  		 			     			  	 
):  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 	 	 			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 	 	 			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	 	 			  		 			     			  	 
    :type start_val: int  		  	   		 	 	 			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	 	 			  		 			     			  	 
    :type commission: float  		  	   		 	 	 			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	 	 			  		 			     			  	 
    :type impact: float  		  	   		 	 	 			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	 	 			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is the function the autograder will call to test your code  		  	   		 	 	 			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	 	 			  		 			     			  	 
    # code should work correctly with either input  		  	   		 	 	 			  		 			     			  	 
    # TODO: Your code here  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Read Data and check for missing data
    if isinstance(orders_file, pd.DataFrame):
        data = orders_file.copy()
    else:
        data = pd.read_csv(orders_file, parse_dates=['Date'])

    # Define start and end dates and range of dates
    start_date = data.index[0]
    end_date =data.index[-1]
    dates = pd.date_range(start_date, end_date)

    # Get stock prices for range of dates and check for missing data, fill forward then backward
    prices = get_data([symbol], dates)
    prices = prices[[symbol]]
    prices['Cash'] = 1.0
    prices.fillna(method="ffill", inplace=True, axis=0)
    prices.fillna(method="bfill", inplace=True, axis=0)
    
    temp_df = data.copy()
    temp_df = temp_df[temp_df['Shares'] != 0].copy()
    temp_df.reset_index(inplace=True)
    temp_df.rename(columns={'index': 'Date'}, inplace=True)
    temp_df['Symbol'] = symbol
    temp_df['Order'] = temp_df['Shares'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
    temp_df['Shares'] = temp_df['Shares'].abs()
    data = temp_df[['Date', 'Symbol', 'Order', 'Shares']]

    trade = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    trade.iloc[0, trade.columns.get_loc('Cash')] = start_val

    # Loop through orders and update portfolio values
    for index, row in data.iterrows():
        date = row['Date']
        shares = row['Shares']
        action = row['Order']
        current_price = prices.loc[date, symbol]

        # If buy, update share of that company and update cash
        if action == "BUY":
            trade.loc[date] += shares
            trade.loc[date, 'Cash'] -= (current_price * shares * (1 + impact) + commission)

        # If sell, update share of that company and update cash
        else:
            trade.loc[date] -= shares
            trade.loc[date, 'Cash'] += (current_price * shares * (1 - impact) - commission)
        
    # Update holdings
    holdings = trade.cumsum()

    # Calculate portfolio values and return
    portfolio_values = holdings * prices
    
    portfolio_values = portfolio_values.sum(axis=1)
    return portfolio_values	  	   		 	 	 			  		 			     			  		  	   		 	 	 			  		 			     			  	 

def port_stats(portvals):
    cr = (portvals[-1] / portvals[0]) - 1
    dr = portvals.pct_change().dropna()
    adr = dr.mean()
    sddr = dr.std()
    sr = (adr / sddr) * np.sqrt(252)
    return cr, adr, sddr, sr
	 	 	 			  		 			     			  	 
def test_code():  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    Helper function to test code  		  	   		 	 	 			  		 			     			  	 
    """  		  	   		 	 	 			  		 			     			  	 
    # this is a helper function you can use to test your code  		  	   		 	 	 			  		 			     			  	 
    # note that during autograding his function will not be called.  		  	   		 	 	 			  		 			     			  	 
    # Define input parameters  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    of = "./orders/orders-02.csv"  		  	   		 	 	 			  		 			     			  	 
    sv = 1000000  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Process orders  		  	   		 	 	 			  		 			     			  	 
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	 	 			  		 			     			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 			  		 			     			  	 
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 			  		 			     			  	 
    else:  		  	   		 	 	 			  		 			     			  	 
        "warning, code did not return a DataFrame"  		  	   		 	 	 			  		 			     			  	 
  		   		 	 	 			  		 			     			  	 
    # Get portfolio stats  		  	   		 	 	 			  		 			     			  	 
    start_date = portvals.index[0] 		  	   		 	 	 			  		 			     			  	 
    end_date = portvals.index[-1] 
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = port_stats(portvals)

    # Get SPY data, calculate SPY stats
    dates = pd.date_range(start_date, end_date)	 
    SPY_portvals = get_data(['$SPX'], dates)['$SPX']
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = port_stats(SPY_portvals)	  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
    # Compare portfolio against $SPX  		  	   		 	 	 			  		 			     			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		 	 	 			  		 			     			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		 	 	 			  		 			     			  	 
    print()  		  	   		 	 	 			  		 			     			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
  		  	   		 	 	 			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	 	 			  		 			     			  	 
    test_code()  		  	   		 	 	 			  		 			     			  	 
