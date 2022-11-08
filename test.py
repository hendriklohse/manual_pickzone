# lst = []
# with open("sku_qty_distribution.csv") as f:
# 	for line in f:
# 		lst.append(line.split(";"))

# print(lst)

# for elt in lst:
# 	for subelt in elt:
# 		# print(subelt)
# 		subelt.replace("\n", "")
# print(lst[1])
#
# with open("sku_qty_distribution_new.csv", "w") as fn:
# 	for elt in lst:
# 		fn.write(elt[0] + "," + elt[1] + "," + elt[2] + "," + elt[3])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#
# df = pd.read_csv("totes_new.csv")
# df.sort_values(by='deadline', inplace=True)
# # print(df.tail())
#
# df.deadline.value_counts().sort_index().plot.bar()
# plt.show()
# #
# timelist = []
# for index, row in df.iterrows():
# 	timelist.append(row['deadline'][11:])
# # print(timelist)
#
# difList = []
# for i in range(len(timelist) - 1):
# 	if timelist[i] != timelist[i+1]:
# 		# difList.append(timelist[i])
# 		difList.append(timelist[i+1])
#
#
# print(difList)
# print(len(difList)) #observed: every 17 minutes a truck leaves with around 275 totes
# print(49*17/60)
# plt.plot(difList)
# plt.show()

#
# import plotly.express as px
# l_values = np.linspace()
# mu1_values = np.linspace()
# mu2_values = np.linspace()
# threshold_result = np.linspace
# df = pd.DataFrame(data=np.random.random((10,10)))
# df = px.data.iris()
# fig = px.parallel_coordinates(df, color="threshold_result",
# 							  dimensions=['l', 'mu1', 'mu2'],
# 							  color_continuous_scale=px.colors.diverging.Tealrose,
# 							  color_continuous_midpoint=2)
# fig.show()