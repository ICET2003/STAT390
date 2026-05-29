# Final Presentation Notes

Presenter: IceT Thaewanarumitkul

## Main Talk Track

1. I tested whether weather variables improve prediction of mental-health treatment and a PCA burnout index.
2. The controlled experiment compared non-weather and weather-augmented feature sets under the same split and model list.
3. The models clearly beat dummy baselines: treatment improved from 0.3915 to about 0.796 weighted F1, and burnout improved from about 0 to about 0.796 R-squared.
4. Weather improved the final metrics only slightly.
5. The treatment model showed the clearest weather signal: wind_gust was the third most important feature, and pressure_hpa also appeared in the top ten.
6. Burnout prediction was dominated by sleep, occupation, day type, and health variables.
7. The conclusion is cautious: weather may matter indirectly, but it was not a major predictor in this dataset.
