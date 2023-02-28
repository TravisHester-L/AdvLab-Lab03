## Data Visualizations

### Mapping

Some plots showing how the error in the fit changes for different combinations of parameters.

Only data for Run 1 and Run 2 are available (brute force computing takes a while). Run 1 is the most different, the rest look similar to Run 2.

* `Run 1 - error.png`: Plots the avg squared error between the data and the fit for points where the 0.1 < error < 1. The hole is to highlight the space containing the closest fit (error < 0.1).

* `Run 2 - error.png`: Same as Run 1. The hole is much smaller because the fit accuracy changes *very* quickly for small changes in each parameter.

* `Run 1 - dError_dbeta.png`: Plots the change in the error with respect to changes in beta for points where that derivative is between -2 and 2.

* `Run 2 - dError_dbeta.png`: Same as Run 1. Much smaller slice again because of such high sensitivity.