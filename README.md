# GMRES and SAM

This is a implementation of GMRES and SAM.

- Tolerance is set to 1e-8. 
- Single precision is used where possible (easier to get good performance on consumer hardware).

Based on:

"Preconditioning Parametrized Linear Systems" by Carr et al.

```
@article{doi:10.1137/20M1331123,
	author = {Carr, Arielle and de Sturler, Eric and Gugercin, Serkan},
	doi = {10.1137/20M1331123},
	eprint = {https://doi.org/10.1137/20M1331123},
	journal = {SIAM Journal on Scientific Computing},
	number = {3},
	pages = {A2242-A2267},
	title = {Preconditioning Parametrized Linear Systems},
	url = {https://doi.org/10.1137/20M1331123},
	volume = {43},
	year = {2021},
	bdsk-url-1 = {https://doi.org/10.1137/20M1331123}}
```

