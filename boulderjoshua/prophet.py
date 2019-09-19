import pandas as pd
from fbprophet import Prophet

input = "2020-08-01"

forecastabbrev10 = pd.read_csv("~/Documents/campgroundpredictions_twoyears.csv")
az = forecastabbrev10.loc[forecastabbrev10['date'] == input]
az
az2=az[['belle_yhat','br_yhat','cw_yhat','hv_yhat','ic_yhat','jr_yhat','ryan_yhat','wt_yhat']]
az3=pd.DataFrame(az2)
az3.columns=['Belle','Black Rock','Cottonwood','Hidden Valley','Indian Cove','Jumbo Rocks','Ryan','White Tank']
az3=pd.melt(az2)
az3=az3.sort_values('value')
az3=az3.reset_index()
bestcamp = (az3['variable'][0])
bcocc = (az3['value'][0])
bestcamp2 = (az3['variable'][1])
bcocc2 = (az3['value'][1])
bestcamp3 = (az3['variable'][2])
bcocc3 = (az3['value'][2])
print('Camps by Likelihood of Availability on',input)
print()
print('Best Camp: ')
print(bestcamp)
print('Predicted Monthly Occupancy: ',bcocc)
print()
print('Next Best Camp: ')
print(bestcamp2)
print('Predicted Monthly Occupancy: ', bcocc2)
print()
print('Third Best Camp: ')
print(bestcamp3)
print('Predicted Monthly Occupancy: ',bcocc3)
print()
