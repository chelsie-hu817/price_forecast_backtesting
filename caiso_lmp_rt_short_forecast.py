# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:09:02 2023

@author: LuZhao
"""
import lightgbm as lgb

from caiso_lmp_rt_forecast import caiso_lmp_rt_forecast


class caiso_lmp_rt_short_forecast(caiso_lmp_rt_forecast):
    def __init__(self, name, price_node):
        """
        Parameters
        ----------
        name : STR
            a description of project name.
        price_node : INT
            used to pull price data.

        Returns
        -------
        caiso lmp forecast client (at rt level)
        rt forecast frame - shorten time period to enhance forecast.


        """
        super().__init__(name, price_node)

    @property
    def YES_ENERGY_TIME_TRANSFORMATIONS(self):
        YES_ENERGY_TIME_TRANSFORMATIONS = super().YES_ENERGY_TIME_TRANSFORMATIONS

        YES_ENERGY_TIME_TRANSFORMATIONS["RTLMP"].extend([2, 3, 4])
        return YES_ENERGY_TIME_TRANSFORMATIONS

    @property
    def TARGETS(self):
        return ['RTLMP']

    @property
    def MODELS(self):
        rt_model = lgb.LGBMRegressor(
            objective='mse',
            num_boost_round=225,
            max_depth=10,
            min_child_weight=30,
            learning_rate=0.06,
            subsample=0.6,
            colsample_bytree=0.2,
            # fit_on_null_values=False,
            # predict_on_null_values=False,
            num_leaves=100)

        return {'RT': rt_model}