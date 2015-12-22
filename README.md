# parallel-crbm
A GPU implementation of CRBM with matlab interface.

## Installation 
### Requirements
* Matlab >= R2012b
* CUDA >= 6.5
* Ubuntu >= 12.04
* gcc/g++ >= 4.6

### Make Install
Just run the make_compile.m file in Matlab. 

## Testing
### cpu mode
start the test_crbmTrain_cpu.m script

### gpu mode
start the test_crbmTrain_gpu.m script

## References
* L. Honglak, R. Grosse, R. Ranganath, and A. Y. Ng, “Unsupervised learning of hierarchical representations with convolutional deep belief networks,” Commun. ACM, vol. 54, no. 10, pp. 95–103, 2011.
* H. Qiao, X. Xi, Y. Li, W. Wu, and F. Li (2014b). Biologically inspired visual model with preliminary cognition and active attention adjustment. IEEE Trans. Syst. Man Cybern. B. doi: 10.1109/TCYB.2014.2377196. [Epub ahead of print].
* H. Qiao, Y. Li, F. Li, X. Xi, and W. Wu, Biologically Inspired Model for Visual Cognition Achieving Unsupervised Episodic and Semantic Feature Learning. IEEE Trans. Cybern. 2015 Sep 18. [Epub ahead of print]
* Y. Li , W. Wu, B. Zhang and F. Li, Enhanced HMAX model with feedforward feature learning for multiclass categorization, Frontiers in Computational Neural Science, 2015 Oct. 07.
