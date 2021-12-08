The purpose of our metric localization leaderboard is to benchmark mapping and localization pipelines. In this scenario, we envision a situation where one or more repeated traversals of the Glen Shields route are used to construct a map offline. Any and all data  from the training sequences may be used to construct a map in any fashion.

Then, during a test sequence, the goal is to perform metric localization between the live sensor data and the pre-built map. Localization approaches may make use of temporal filtering and can leverage the IMU if desired but GPS information will not be available. The goal of this benchmark is to simulate localizing a vehicle in real-time and as such methods may not use future sensor information in an acausal manner.

Our goal is to support both global and relative map structures. Only one of the training sequences will specified as the map sequence used by the benchmark. For 3D localization, users must choose either the lidar or camera as the reference sensor. For 2D localization, only the radar frames are used as a reference. For each (camera|lidar|radar) frame $s_2$ in the test sequence, users will specify the ID (timestamp) of the (camera|lidar|radar) frame $s_1$ in the map sequence that they are providing a relative pose with respect to: $\hat{\mathbf{T}}_{s_1,s_2}$. We then compute root-mean squared error (RMSE) values for the translation and rotation as follows:

$$
\begin{align*}
	\mathbf{T}_e &=  \carrot({T}_{a,s_1} \mathbf{T}_{s_1,s_2} \hat{\mathbf{T}}_{s_1,s_2}^{-1} \mathbf{T}_{a,s_1}^{-1} = \begin{bmatrix} \mathbf{C}_e & \mathbf{r}_e \\ \mathbf{0}^T & 1  \end{bmatrix} \\
	\mathbf{r}_e &= \begin{bmatrix} x_e & y_e & z_e \end{bmatrix}^T \\
	\phi_e &= \arccos \left( \frac{\text{tr}~\mathbf{C}_e - 1}{2}  \right)
\end{align*}
$$

where $\mathbf{T}_{s_1,s_2}$ is the known ground truth pose, and $\mathbf{T}_{a,s_1}$ is the calibrated transform from the sensor frame to the applanix frame (x-right, y-forwards, z-up). $x_e, y_e, z_e$ are then the lateral, longitudinal, and vertical errors respectively. We calculate RMSE values for $x_e, y_e, z_e, \phi_e$.

For each test sequence, users will provide a space-seperated text file of K x (14|50) values. The first column is the timestamp of the test frame, the second column is the timestamp of the user-specified reference frame in the map sequence. The following 12 values correspond to the upper 3x4 component of $\mathbf{T}}_{s_1,s_2}$ stored in row-major order. Users also have the option of providing $6 \times 6$ covariance matrices $\mbs{\Sigma}_i$ for each localization estimate. The entire covariance matrix must be unrolled into 36 values (row-major order) and appended to each row, for a total of 50 values per row.