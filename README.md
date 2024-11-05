# An ensemble score filter for tracking high-dimensional nonlinear dynamical systems
Code repository for the paper:  
**Ensemble Score Filter for Tracking High-dimensional Nonlinear Dynamical System**  
[Feng Bao](https://www.math.fsu.edu/~bao/), [Zezhong Zhang](https://www.ornl.gov/staff-profile/zezhong-zhang), [Guannan Zhang](https://sites.google.com/view/guannan-zhang)
[paper](https://www.sciencedirect.com/science/article/pii/S0045782524007023)
## Usage
1. [`data`](https://github.com/zezhongzhang/EnSF/tree/main/data) contains the initial state of L96 model and the random shock profile.
   * [`gen_shock.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/data/gen_shock.ipynb) generates the random shock profiles.
   * [`gen_state_init.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/data/gen_state_init.ipynb) generates the initial states for L96 model with different dimensions.
   * Generate the initial state first before running the one-million-dimensional problem.
2. [`fine_tune_EnSF`](https://github.com/zezhongzhang/EnSF/tree/main/fine_tune_EnSF) is the folder for fine-tuning hyperparameters of EnSF ([`fine_tune_LETKF`](https://github.com/zezhongzhang/EnSF/tree/main/fine_tune_LETKF) for LETKF).
   * [`gen_filter_param.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/gen_filter_param.ipynb) generates parameter combinations in [`para_EnSF.csv`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/param_EnSF.csv)
   * [`param_problem.csv`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/param_problem.csv) is the filtering problem setting for fine-tuning.
   * [`run_all_EnSF.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/run_all_EnSF.ipynb) runs all combinations of parameters stored in `results`.
   * [`results/result_summary.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/results/result_summary.ipynb) aggregates all the results to [`results/final_rmse_EnSF.csv`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/results/final_rmse_EnSF.csv) and [`results/rmse_EnSF.csv`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/results/rmse_EnSF.csv)
   * [`to_grid.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/fine_tune_EnSF/results/to_grid.ipynb) generates 2D grid plot for fine-tuning results.
3. [`run_all_EnSF`](https://github.com/zezhongzhang/EnSF/tree/main/run_all_EnSF) is the folder for repetitive runs of different filtering settings and parameter combinations ([`run_all_LETKF`](https://github.com/zezhongzhang/EnSF/tree/main/run_all_LETKF) for LETKF).
   * [`param_EnSF.csv`](https://github.com/zezhongzhang/EnSF/blob/main/run_all_EnSF/param_EnSF.csv) and [`param_problem.csv`](https://github.com/zezhongzhang/EnSF/blob/main/run_all_EnSF/param_problem.csv) are user input settings for filtering problem and filter.
   * [`gen_param_all.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/run_all_EnSF/gen_param_all.ipynb) generates all the problem/filter combinations in [`param_combined.csv`](https://github.com/zezhongzhang/EnSF/blob/main/run_all_EnSF/param_combined.csv) with different random seeds for repeated experiments.
   * [`run_all_EnSF.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/run_all_EnSF/run_all_EnSF.ipynb) runs all problem/filter combinations and stores the results in `result`
4. [`run_single_EnSF`](https://github.com/zezhongzhang/EnSF/tree/main/run_single_EnSF) is the folder for a single run of the filter ([`run_single_LETKF`](https://github.com/zezhongzhang/EnSF/tree/main/run_single_LETKF) for LETKF).
   * [`run_single_EnSF.ipynb`](https://github.com/zezhongzhang/EnSF/blob/main/run_single_EnSF/run_single_EnSF.ipynb) runs the problem/filter combination in [`param_combined.csv`](https://github.com/zezhongzhang/EnSF/blob/main/run_single_EnSF/param_combined.csv) and stores the results in `result`


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
