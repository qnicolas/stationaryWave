For a nice formulation of the hydrostatic primitive equations in sigma coordinates: see page 41 of
https://ethz.ch/content/dam/ethz/special-interest/usys/iac/iac-dam/documents/edu/courses/weather_and_climate_models/FS2023/Slides/03b_WCM_VertCoord.pdf

All fields are separated into a basic state (overbars) and a perturbation (would usually have primes, but we omit them here). For example, the full velocity field is $\overline{\mathbf{u}_i} + \mathbf{u}_i$.

Vertical staggering: $\dot{\sigma}$ and $\Phi$ on full levels, $\mathbf{u},T$ on half levels. Index $i=0$ is at the top of the domain; $i=N$ is at the surface. See sketch below:

![[sigmalevsketch.png|600]]

Boundary conditions: $\dot{\sigma}(\sigma=0) = \dot{\sigma}_0 = 0$,  $\dot{\sigma}(\sigma=1) = \dot{\sigma}_N = 0$, $\Phi(\sigma=1) = \Phi_N = \Phi_{sfc}$ 

Momentum equations: 
$$\begin{array}{cll}
\dfrac{\partial \mathbf{u}_i}{\partial t} 
&+& \overline{\mathbf{u}_i}\cdot\nabla\mathbf{u}_i + \mathbf{u}_i\cdot\nabla\overline{\mathbf{u}_i}\\ 
&+& \dfrac{1}{2\Delta\sigma}\left[\overline{\dot{\sigma}_i}(\mathbf{u}_{i+1} - \mathbf{u}_i) + \overline{\dot{\sigma}_{i-1}}(\mathbf{u}_i - \mathbf{u}_{i-1})\right]
+ \dfrac{1}{2\Delta\sigma}\left[\dot{\sigma}_i\overline{(\mathbf{u}_{i+1} - \mathbf{u}_i)} + \dot{\sigma}_{i-1}\overline{(\mathbf{u}_i - \mathbf{u}_{i-1})}\right]\\ 
&+& f\mathbf{k}\times\mathbf{u}_i\\ 
&=& -\left\{ \nabla\dfrac{\Phi_i + \Phi_{i-1}}{2} + \epsilon_i \mathbf{u}_i + \nu\nabla^4\mathbf{u}_i
+ R\overline{T}\nabla\ln p_s + RT\nabla\overline{\ln p_s}\right\}
\end{array}$$

Thermodynamic equations: 
$$\begin{array}{cll}
\dfrac{\partial T_i}{\partial t} 
&+& \overline{\mathbf{u}_i}\cdot T_i + \mathbf{u}_i\cdot\nabla\overline{T_i}\\ 
&+& \dfrac{1}{2\Delta\sigma}\left[\overline{\dot{\sigma}_i}(T_{i+1} - T_i) + \overline{\dot{\sigma}_{i-1}}(T_i - T_{i-1})\right]
+ \dfrac{1}{2\Delta\sigma}\left[\dot{\sigma}_i\overline{(T_{i+1} - T_i)} + \dot{\sigma}_{i-1}\overline{(T_i - T_{i-1})}\right]\\ 
&-& \dfrac{\kappa}{(i-1/2)\Delta\sigma}\left[\overline{T}_i\dfrac{\dot\sigma_i + \dot\sigma_{i-1}}{2} + T_i\dfrac{\overline{\dot\sigma_i + \dot\sigma_{i-1}}}{2}\right]\\
&-& \kappa \overline{T_i}\dfrac{\partial \ln p_s }{\partial t} 
- \kappa T_i \overline{\mathbf{u}_i}\cdot\nabla\overline{\ln p_s}
- \kappa \overline{T_i} \mathbf{u}_i\cdot\nabla\overline{\ln p_s}
- \kappa \overline{T_i} \overline{\mathbf{u}_i}\cdot\ln p_s 
 \\
&=& -\left\{ \epsilon_i T_i + \nu\nabla^4T_i \right\}\\
&&+Q_{\mathrm{diab},i}
\end{array}$$

Prognostic equation for $\ln(p_s)$: $\dfrac{\partial \ln(p_s)}{\partial t} = - \Delta\sigma\left[\nabla\cdot\left(\sum_{i=1}^{N}\mathbf{u}_i\right) + \left(\sum_{i=1}^{N}\mathbf{u}_i\right)\cdot\nabla\overline{\ln(p_s)} + \left(\sum_{i=1}^{N}\overline{\mathbf{u}_i}\right)\cdot\nabla\ln(p_s)\right]$ 

Diagnosis of $\dot{\sigma}$: $\dot{\sigma}_i = -i\Delta\sigma\dfrac{\partial \ln(p_s)}{\partial t} - \Delta\sigma\left[\nabla\cdot\left(\sum_{j=1}^{i}\mathbf{u}_j\right) + \left(\sum_{j=1}^{i}\mathbf{u}_j\right)\cdot\nabla\overline{\ln(p_s)} + \left(\sum_{j=1}^{i}\overline{\mathbf{u}_j}\right)\cdot\nabla\ln(p_s)\right]$

Diagnosis of $\Phi$: $\dfrac{\partial \Phi}{\partial \sigma} = -\dfrac{RT}{\sigma}$ so $\dfrac{\Phi_{i}-\Phi_{i-1}}{\Delta \sigma} = -\dfrac{RT_i}{(i-1/2)\Delta\sigma}$ with $\Phi_N = \Phi_{sfc}$ 
Hence $\Phi_i = \Phi_{sfc} + R\sum_{j=i+1}^{N} \dfrac{T_j}{j-1/2}$ 

