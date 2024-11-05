# An ensemble score filter for tracking high-dimensional nonlinear dynamical systems
Code repository for the paper:  
**Ensemble Score Filter for Tracking High-dimensional Nonlinear Dynamical System**  
[Feng Bao](https://www.math.fsu.edu/~bao/), [Zezhong Zhang](https://www.ornl.gov/staff-profile/zezhong-zhang), [Guannan Zhang](https://sites.google.com/view/guannan-zhang)

## Usage
1. [`data`]() contains the initial state of L96 model and the random shock profile.
    * [`gen_shock.ipynb`]() generates the random shock profiles.
    * [`gen_state_init.ipynb`]() generates the initial states for L96 model with different dimensions.
2. [`fine_tune_EnSF`]() and [`fine_tune_LETKF`]() are the folders for fine-tuning hyperparamters.
    * [`gen_filter_param.ipynb`]() generate parameter combinations [`para_EnSF.csv`]()
    * [`param_problem.csv`]() is the filtering problem setting for fine-tuning.
    * [`run_all_EnSF.ipynb`]() run all comnations of paramters stored in `results`
4. [`run_single_EnSF`]() and [`run_single_LETKF`]() are the folders for a single run of the filter.
5. [`run_all_EnSF`]() and [`run_all_LETKF`]() are the folders for repetitive run of different filtering setting and differnt paramter combinations.


## Citation
If you  find the idea or code of this paper useful for your research, please consider citing us:

```bibtex
@article{bao2024ensemble,
  title={An ensemble score filter for tracking high-dimensional nonlinear dynamical systems},
  author={Bao, Feng and Zhang, Zezhong and Zhang, Guannan},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={432},
  pages={117447},
  year={2024},
  publisher={Elsevier}
}
```
