## LWdn
Fits a regression model to atmospheric downward longwave radiation data and uses it to predict new values

### Background
The Earth atmosphere has a downward radiation component in the thermal radiation wavelength range. There are situations where it would be desirable to estimate the amount of this downward longwave radiation from generic climatological measurements conducted at the ground surface. There are already a number of empirical equations and detailed numerical models for this, but the empirical equations are fundamentally related to the data sets that were used to fit the mode, and the detailed numerical models can be difficult to use for average users.

The *LWdn* module implements a *fit()* function, that can be used to fit certain models to atmospheric downward longwave radiation (LWdn) data, using commonly available meteorological measurements as inputs. The *predict()* function can be used after that to give the model new input data and predict the LWdn values from these values.
