import os


path = 'train'
def batch_rename(path):
    count = 0
    for i in os.listdir(path):
    	if len(i) == 6:
    		os.rename(os.path.join(path, i), os.path.join(path, '0' + i))
    	'''
        new_i = str(count)
        print os.path.join(path, i)
        os.rename(os.path.join(path, i), os.path.join(path, new_fname))
        count = count + 1 
        '''

batch_rename(path)