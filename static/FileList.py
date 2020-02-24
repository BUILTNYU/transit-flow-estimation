# This class is used to create a file list of all the .gz data filenames

import os

class FileList():
	"""Create a object of raw data file path list."""

	def __init__(self,path):

		# Path of file to open.
		# File path = dataDirectory + '201607/' + 'SPTCC-20160701.csv.gz'

		self.file_list=[]

		# Date format: 20160709
		self.date = []

		self.sys_dir = os.path.join(os.path.expanduser('~'), path, 'smartcard', '')
		self.data_dir = os.path.join(os.path.expanduser('~'), path, 'smartcard', 'data', '')
		
		months = ['201607', '201608', '201609']
		for mon in months:
			for day in range(31):
			
				if not (mon == '201609' and day == 30): # September has no 31th day
					
					if (day + 1) <= 9:
						tempstr = '0'
					else:
						tempstr = ''

					self.file_list.append(os.path.join(self.data_dir + mon, 'SPTCC-' + mon + tempstr + str(day + 1) + '.csv.gz'))
					
					if day < 9:
						self.date.append(mon + '0' + str(day + 1))
					else:
						self.date.append(mon + str(day + 1))