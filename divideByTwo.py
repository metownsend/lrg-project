# Function to convert decimal to binary
# import sys
# __init__.py
#!/usr/bin/env python

def divideBy2(decNumber):

	from pythonds.basic.stack import Stack
	import numpy as np

	np.vectorize(decNumber)  
	remstack = Stack()
	
	if decNumber == 0: return "0"
	
	while decNumber > 0:
		rem = decNumber % 2
		remstack.push(rem)
		decNumber = decNumber // 2
		
	binString = ""
	while not remstack.isEmpty():
		binString = binString + str(remstack.pop())
			
	return binString
