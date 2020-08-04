print(tot_return)

print(pd.DataFrame.from_dict(annualreturns, orient='index'))
print(pd.DataFrame.from_dict(returns, orient='index') )
print(pd.DataFrame.from_dict(lorret, orient='index') )
print(pd.DataFrame.from_dict(timeret, orient='index') )

print(sharpe_ratio)
print(vwr)

print(drawdown['max']['len'])
print(drawdown['max']['drawdown'])
print(drawdown['max']['moneydown'])

print(pd.DataFrame.from_dict(timedd, orient='index') )