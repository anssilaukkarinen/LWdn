
# LWdn

Calculate hourly atmospheric downward long-wave radiation values for future climatic conditions using input data and machine learning

## Background

The climatic processes in typical building applications can be categorized as follows:

- Outdoor air temperature and humidity
- Wind and precipitation
- Solar radiation
- Long-wave radiation

The data from the RASMI project contains the following variables. The variable descriptions are in line with the World Meteorological Organization WMO guidelines.

- Time as in normal Finnish time (winter time)
- Outdoor air temperature at 2 m heigh from ground surface, degC, instantaneous value
- Outdoor air relative humidity at 2 m height from ground surface (with respect to liquid water), %, instantaneous value
- Wind direction, positive clock-wise from north, deg, instantaneous value. For example, wind direction of 90 deg means that wind is blowing from the east.
- Wind speed at 10 m height, instantaneous scalar value, m/s
- Total precipitation to a horizontal surface (snow is not removed), mm/h, cumulative value from previous time point
- Global horizontal solar radiation, W/m2, average value after the previous time point. For example, value at 15:00 means average value between 14:00 and 15:00.
- Diffuse horizontal solar radiation, W/m2, average value after the previous time point
- Direct normal solar radiation, W/m2, average value after the previous time point, value is for the surface normal to sun's ray


The dataset contain full hourly time series for 30 years time period between 1989 and 2018 for four locations of Vantaa, Jokioinen, Jyväskylä and Sodankylä. The past time series are based on measurement data from the weather stations that the Finnish Meteorological Institute (FMI) operates. In addition, full 30 year hourly time series were created also for future climatic conditions surrounding years 2030, 2050 and 2050, based on emission scenarios of RCP2.6, RCP4.5 and RCP8.5. These were created by modifying the past measurement data using climate model predictions from IPCC. The total number of 30 year hourly data sets was 4 locations * (1 + 3*3) conditions = 40 datasets.

Long-wave radiation is generally the one that is most often not available in climatic datasets and it was also not part of the original RASMI data. From outdoor climatic variables perspective, the two long-wave radiation components are atmospheric downward long-wave radiation (LWdn) and the ground upward long-wave radiation (LWup). Of these the atmospheric downward long-wave radiation LWdn was added to the RASMI dataset afterwards. All code related to this process is in this GitHub repository and the main steps of the process is explained in the next section.

## Input data and LWdn for past climatic conditions

The Finnish Meteorological Institute FMI measures atmospheric downward long-wave radiation in some of the weather stations it operates. The length of measurement data and the amount of missing values however varies, so at the precent moment it was not possible to formulate full 30 year time series from measured data only.

The ERA5 reanalysis data in the Copernicus Climate Data Store contains hourly values for atmospheric downward long-wave radiation. This was compared to the FMI weather station data and although there was some differences, on average the ERA5 reanalysis data matched the measured values well. As a result, it was decided to use the hourly ERA5 renalaysis data for the past climate of 1989-2018.

At the time being, the ERA5 reanalysis data does not contain values for future climatic conditions and the IPCC climate model results do not include hourly data. This means that other approach is needed to generate hourly LWdn values for future climatic conditions.

## LWdn values for future climatic conditions

There are various semi-empirical model available in the literature that can be used to estimate the hourly atmospheric downward long-wave radiation from regular weather station measurements. A comparison of these semi-empirical methods was done by Jokela ([2018](https://urn.fi/URN:NBN:fi:tty-201811262762)), where the model presented by Mundt-Petersen and Wallentén ([2014](https://portal.research.lu.se/en/publications/methods-for-compensate-lack-of-climate-boundary-data)) was selected. Besides outdoor air temperature and dew point temperature, the model also included the clearness index describing the atmospheric conditions, and it was defined for Swedish climatic conditions, which is close to the Finnish conditions.

During the RAMI project it was decided to test machine learning for predicting atmospheric downward long-wave radiation for future climatic conditions and compare the results to the previously selected semi-empirical model. The machine learning (ML) workflow was as follows:

- Read in RASMI data and ERA5 LWdn data
- Train different ML models using 27 out of 30 years data. This included ML model hyperparameter optimization by random search cross-validation using the [sklearn](https://scikit-learn.org) library.
- Use the fitted ML models and the semi-empirical model to predict the hourly LWdn values for the remaining 3 out of 30 years of data. Use also the sklearn DummyRegressor() function as a baseline.
- Select the model that had the lowest Mean Absolute Error (MAE) across all four studied locations (Vantaa, Jokioinen, Jyväskylä, Sodankylä). This was the LGBMRegressor().
- Use the selected model to predict the hourly LWdn values for all future climatic conditions. The model fitting and the predictions are done separately for each studied location.


## Evaluation of results

The evaluation of results was done in the following ways:

- Visual inspection of results using time series plots and scatter plots
- Comparison of the ERA5 atmospheric downward long-wave radiation data to measured data from Finnish Meteorological Institute
- Comparison of properties and results from semi-empirical models to each other in a previous project
- Comparison of results from machine learning models to each other and to semi-empirical model for past climate ERA5 data
- Comparison of monthly means of predicted future LWdn data to climate model (CMIP6) predictions.

For the last part, the percent change in monthly mean values were determined for multiple climate models. These results were then used as a reference distribution, to which the predictions from the machine learning calculations were compared to. The monthly means of the ML model predictions were close to the monthly means of the climate model predictions.

As a summary, it was considered that using machine learning with reanalysis data allowed to create usable predicted values of hourly atmospheric downward long-wave radiation for future climatic conditions. These values are now provided as part of the RASMI data under the same [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

## Final report of the RAMI project

The final report of the RAMI project (Rakentamisen mitoitussäät, eng. Design weather data for Finnish climatic conditions) is available here:

Laukkarinen, A., Jokela, T., Vinha, J., Pakkala, T., Lahdensivu, J., Lestinen, S., Jokisalo, J., Kosonen, R., Lindfors, A., Ruosteenoja, K. & Jylhä, K. (2022) Vaipparakenteiden rakennusfysikaalisen toimivuuden ja huonetilojen kesäaikaisen jäähdytystehontarpeen mitoitusolosuhteet : RAMI-hankkeen loppuraportti. 	Tampereen yliopisto. Rakennustekniikka. Tutkimusraportti 3. ISSN 2669-8838. ISBN 978-952-03-2438-4. https://urn.fi/URN:ISBN:978-952-03-2438-4. 192 p. [In Finnish, Abstract in English]
