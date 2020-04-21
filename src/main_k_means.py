import sys
from LoadData import * 
from k_means import * 
from evaluation import * 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "[usage] <data-file> <ground-truth-file>"
        exit(1) 
    
    dataFilename = sys.argv[1]
    groundtruthFilename = sys.argv[2]
    
    data = loadPoints(dataFilename) 
    groundtruth = loadClusters(groundtruthFilename) 
    
    nDim = len(data[0]) 
   
    K = 3  # Suppose there are 2 clusters
    print 'K=',K

    # use the first two data points as initial cluster centers
    centers = []
    for i in range(K):
        centers.append(data[i])


    # get clusterID, list
    results = kmeans(data, centers) 
    
    
    #figure
    import numpy as np
    from matplotlib import pyplot as plt
    data2=np.array(data)
    kmeans = kmeans(data, centers) 
    plt.scatter(data2[:, 0], data2[:, 1], c=groundtruth, s=40, cmap='viridis');
  

    res_Purity = purity(groundtruth, results) 
    res_NMI = NMI(groundtruth, results) 
    
    print "Purity =", res_Purity
    print "NMI = ", res_NMI
  
   # from matplotlib import pyplot as plt
  #  import numpy as np
  #  data2=np.array(data)
  #  d =plt.figure(1)
  #  plt.scatter(data2[:,0],data2[:,1],20,results)
  #  d.show()
