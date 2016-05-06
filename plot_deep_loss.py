lines=open('TPM_whitened_data_3layer_deep_loss_check.txt').readlines()
X=[]
Y=[]
Y_test=[]
for line in lines:
	splits=line.replace('\n','').split('\t')
	#print splits
	index=int(splits[0])
	X.append(index)
	Y.append(float(splits[1]))
	Y_test.append(float(splits[2]))
#print lines
import matplotlib.pyplot as plt
plt.plot(X,Y, label='Train',linestyle='--')
plt.plot(X,Y_test, label='Test')
plt.legend()
plt.show()
