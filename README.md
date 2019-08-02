# cla-to-mcr
By Pulin Agrawal

This project develops a method for conversion of high-dimensional sparse representations to Modular Composite Representation (Snaider et. al., 2013; Snaider & Franklin, 2014) vectors. This is roughly inspired by the random projection technique. Thus, it also serves to preserve distances between two vectors after conversion. The findings were published in a paper (Agrawal et. al., 2018)

For reverse translation
 Probing can be used to probe if the given MCR vector has some set of Projection MCR vector vectors. by placing 1's on the positions of those vectors we can retrieve the CLA vector.

Sparse binary vectors are noise robust. 
MCR vectors provide composition capabilities.

Possible Applications
* Sparse PCA
* Sparse Composite Quantization

## References

Agrawal, P., Franklin, S. & Snaider, J. (2018). [Sensory Memory For Grounded Representations in a Cognitive Architecture](http://ccrg.cs.memphis.edu/assets/papers/Agrawal-2018-sensory-memory-for-grounded-rep-ACS.pdf). Sixth Annual Conference on Advances in Cognitive Systems. 

Snaider, J., Franklin, S., Strain, S., & George, E. O. (2013). [Integer sparse distributed memory: Analysis and results](). Neural Networks, 46, 144-153.

Snaider, J., & Franklin, S. (2014). [Modular Composite Representation](http://ccrg.cs.memphis.edu/assets/papers/2014/MCR%20paper%20final%20review2.pdf). Cognitive Computation, 6(3), 510-527. doi: 10.1007/s12559-013-9243-y

