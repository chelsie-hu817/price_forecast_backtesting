# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:07:02 2023

@author: lzhao
"""
import lightgbm as lgb
import numpy as np

from caiso_price_forecast import DATATYPE_DALMP, DATATYPE_RTLMP
from caiso_price_forecast import DATATYPE_RTLOAD, DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, DATATYPE_WIND_DA

from caiso_lmp_forecast import caiso_lmp_da_forecast, define_net_export_data
from caiso_lmp_forecast import DATATYPE_DA_EXP_MW, DATATYPE_DA_IMP_MW, DATATYPE_RT_EXP_MW, DATATYPE_RT_IMP_MW

# YesEnergy datatypes.

DATATYPE_SOLAR_FORECAST_RT = "SOLARFCST_HOURLY_CONS"
DATATYPE_WIND_FORECAST_RT = "WINDFCST_HOURLY_CONS"

DATATYPE_SOLAR_GEN_RT = 'GENERATION_SOLAR_CURRENT'
DATATYPE_WIND_GEN_RT = 'GENERATION_WIND_CURRENT'

DATATYPE_DALOAD = 'DALOAD'
DATATYPE_LOAD_FORECAST = 'LOAD_FORECAST'


class caiso_lmp_rt_forecast(caiso_lmp_da_forecast):
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
        rt forecast frame - next hour to end of day/24 hour window depends on if da market clear


        """
        super().__init__(name, price_node)

    @property
    def YES_ENERGY_ITEMS(self):

        # YesEnergy price objects.
        PRICE_NODE = self.PRICE_NODE

        # YesEnergy load objects.
        ZONE_CAISO = 10000328798
        ZONE_PGE = 10000328796
        ZONE_SCE = 10000328795
        ZONE_SDGE = 10000328797
        ZONE_VEA = 10002423972

        # YesEnergy generation objects.
        ZONE_NP15 = 10002494909
        ZONE_SP15 = 10002494908

        INTERFACE_CAISO = 10002496439
        INTERFACE_PGE = 10002496442
        INTERFACE_SCE = 10002496441
        INTERFACE_SDGE = 10002496443

        YES_ENERGY_ITEMS = {
            # Energy prices.
            "DALMP": (DATATYPE_DALMP, PRICE_NODE),
            "RTLMP": (DATATYPE_RTLMP, PRICE_NODE),

            # Load forecasts.
            "CAISO_LOAD_FORECAST": (DATATYPE_LOAD_FORECAST, ZONE_CAISO),
            "PGE_LOAD_FORECAST": (DATATYPE_LOAD_FORECAST, ZONE_PGE),
            "SCE_LOAD_FORECAST": (DATATYPE_LOAD_FORECAST, ZONE_SCE),
            "SDGE_LOAD_FORECAST": (DATATYPE_LOAD_FORECAST, ZONE_SDGE),
            "VEA_LOAD_FORECAST": (DATATYPE_LOAD_FORECAST, ZONE_VEA),

            # DA load.
            "CAISO_DALOAD": (DATATYPE_DALOAD, ZONE_CAISO),
            "PGE_DALOAD": (DATATYPE_DALOAD, ZONE_PGE),
            "SCE_DALOAD": (DATATYPE_DALOAD, ZONE_SCE),
            "SDGE_DALOAD": (DATATYPE_DALOAD, ZONE_SDGE),
            # "VEA_DALOAD": (DATATYPE_DALOAD, ZONE_VEA), VEA da load comes back empty

            # Actual load.
            "CAISO_RTLOAD": (DATATYPE_RTLOAD, ZONE_CAISO),
            "PGE_RTLOAD": (DATATYPE_RTLOAD, ZONE_PGE),
            "SCE_RTLOAD": (DATATYPE_RTLOAD, ZONE_SCE),
            "SDGE_RTLOAD": (DATATYPE_RTLOAD, ZONE_SDGE),
            "VEA_RTLOAD": (DATATYPE_RTLOAD, ZONE_VEA),

            # Scheduled renewable generation.
            "NP15_GENERATION_SOLAR_DA_SCHEDULE": (DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, ZONE_NP15),
            "NP15_WIND_DA": (DATATYPE_WIND_DA, ZONE_NP15),
            "SP15_GENERATION_SOLAR_DA_SCHEDULE": (DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, ZONE_SP15),
            "SP15_WIND_DA": (DATATYPE_WIND_DA, ZONE_SP15),

            # Actual renewable generation.
            "NP15_GENERATION_SOLAR_CURRENT": (DATATYPE_SOLAR_GEN_RT, ZONE_NP15),
            "NP15_GENERATION_WIND_CURRENT": (DATATYPE_WIND_GEN_RT, ZONE_NP15),
            "SP15_GENERATION_SOLAR_CURRENT": (DATATYPE_SOLAR_GEN_RT, ZONE_SP15),
            "SP15_GENERATION_WIND_CURRENT": (DATATYPE_WIND_GEN_RT, ZONE_SP15),

            # Forecasted renewable generation.
            "NP15_SOLARFCST_HOURLY_CONS": (DATATYPE_SOLAR_FORECAST_RT, ZONE_NP15),
            "NP15_WINDFCST_HOURLY_CONS": (DATATYPE_WIND_FORECAST_RT, ZONE_NP15),
            "SP15_SOLARFCST_HOURLY_CONS": (DATATYPE_SOLAR_FORECAST_RT, ZONE_SP15),
            "SP15_WINDFCST_HOURLY_CONS": (DATATYPE_WIND_FORECAST_RT, ZONE_SP15),

            # Import/Export.
            'CAISO_DA_IMP_MW': (DATATYPE_DA_IMP_MW, INTERFACE_CAISO),
            'CAISO_DA_EXP_MW': (DATATYPE_DA_EXP_MW, INTERFACE_CAISO),
            'PGE_DA_IMP_MW': (DATATYPE_DA_IMP_MW, INTERFACE_PGE),
            'PGE_DA_EXP_MW': (DATATYPE_DA_EXP_MW, INTERFACE_PGE),
            'SCE_DA_IMP_MW': (DATATYPE_DA_IMP_MW, INTERFACE_SCE),
            'SCE_DA_EXP_MW': (DATATYPE_DA_EXP_MW, INTERFACE_SCE),
            'SDGE_DA_IMP_MW': (DATATYPE_DA_IMP_MW, INTERFACE_SDGE),
            'SDGE_DA_EXP_MW': (DATATYPE_DA_EXP_MW, INTERFACE_SDGE),

            'CAISO_RT_IMP_MW': (DATATYPE_RT_IMP_MW, INTERFACE_CAISO),
            'CAISO_RT_EXP_MW': (DATATYPE_RT_EXP_MW, INTERFACE_CAISO),
            'PGE_RT_IMP_MW': (DATATYPE_RT_IMP_MW, INTERFACE_PGE),
            'PGE_RT_EXP_MW': (DATATYPE_RT_EXP_MW, INTERFACE_PGE),
            'SCE_RT_IMP_MW': (DATATYPE_RT_IMP_MW, INTERFACE_SCE),
            'SCE_RT_EXP_MW': (DATATYPE_RT_EXP_MW, INTERFACE_SCE),
            'SDGE_RT_IMP_MW': (DATATYPE_RT_IMP_MW, INTERFACE_SDGE),
            'SDGE_RT_EXP_MW': (DATATYPE_RT_EXP_MW, INTERFACE_SDGE),

        }

        return YES_ENERGY_ITEMS

    @property
    def YES_ENERGY_TIME_TRANSFORMATIONS(self):
        YES_ENERGY_TIME_TRANSFORMATIONS = {
            "DALMP": [24, 48],
            "RTLMP": [24, 48],

            # Actual load.
            "CAISO_RTLOAD": [24, 168],
            "PGE_RTLOAD": [24, 168],
            "SCE_RTLOAD": [24, 168],
            "SDGE_RTLOAD": [24, 168],
            "VEA_RTLOAD": [24, 168],

            "CAISO_LOAD_FORECAST": [24, 168],
            "PGE_LOAD_FORECAST": [24, 168],
            "SCE_LOAD_FORECAST": [24, 168],
            "SDGE_LOAD_FORECAST": [24, 168],
            "VEA_LOAD_FORECAST": [24, 68],

            "CAISO_LOAD_FORECAST_RESIDUAL": [24, 168],
            "PGE_LOAD_FORECAST_RESIDUAL": [24, 168],
            "SCE_LOAD_FORECAST_RESIDUAL": [24, 168],
            "SDGE_LOAD_FORECAST_RESIDUAL": [24, 168],
            "VEA_LOAD_FORECAST_RESIDUAL": [24, 168],

            # RT net export
            **{f'{zone}_RT_NET_EXP_MW': [24, 168] for zone in ['CAISO', 'PGE', 'SCE', 'SDGE']},
        }
        return YES_ENERGY_TIME_TRANSFORMATIONS

    @property
    def BASE_FEATURES(self):

        base_features = [

                            "DALMP",

                            # DA load.
                            "CAISO_DALOAD",
                            "PGE_DALOAD",
                            "SCE_DALOAD",
                            "SDGE_DALOAD",

                            "CAISO_LOAD_FORECAST",
                            "PGE_LOAD_FORECAST",
                            "SCE_LOAD_FORECAST",
                            "SDGE_LOAD_FORECAST",
                            "VEA_LOAD_FORECAST",

                            "NP15_GENERATION_SOLAR_DA_SCHEDULE",
                            "NP15_WIND_DA",
                            "SP15_GENERATION_SOLAR_DA_SCHEDULE",
                            "SP15_WIND_DA",

                            'NP15_GENERATION_SOLAR_CURRENT_OR_FORECAST',
                            'NP15_GENERATION_WIND_CURRENT_OR_FORECAST',
                            'SP15_GENERATION_SOLAR_CURRENT_OR_FORECAST',
                            'SP15_GENERATION_WIND_CURRENT_OR_FORECAST',
                        ] + [f'{zone}_DA_NET_EXP_MW' for zone in ['CAISO', 'PGE', 'SCE', 'SDGE']]

        return base_features

    @property
    def TARGETS(self):
        return [self.CLIPPED_DART]

    def define_data(self, yes_energy_data):
        # def define_data(self, yes_energy_data, weather_data):
        """
        Parameters
        ----------
        yes_energy_data : DICT
            dict of yes energy data {yes_energy_item:data} .
        weather_data : DICT
            dict of weather data .

        Returns
        -------
        formated input and target data.

        """
        calendar_ts = yes_energy_data['DALMP'][['DATETIME']].copy()  ##use dalmp as example

        calendar_data = self.define_calendar_data(calendar_ts)

        # Since the scheduled datatypes may not be available in time for the model to use them,
        #   we use layering to set up the corresponding forecasts as a backup.
        for zone in ("NP15", "SP15"):
            for scheduled, forecasted in (
                    (DATATYPE_SOLAR_GEN_RT, DATATYPE_SOLAR_FORECAST_RT),
                    (DATATYPE_WIND_GEN_RT, DATATYPE_WIND_FORECAST_RT),
            ):
                # Pop the un-layered time series objects.
                scheduled_name = f"{zone}_{scheduled}"
                forecasted_name = f"{zone}_{forecasted}"
                scheduled_ts = yes_energy_data.pop(scheduled_name)
                forecasted_ts = yes_energy_data.pop(forecasted_name).copy()

                scheduled_ts.rename(columns={'AVGVALUE': 'AVGVALUE_SCHEDULED'}, inplace=True)
                forecasted_ts.rename(columns={'AVGVALUE': 'AVGVALUE_FORECASTED'}, inplace=True)

                # Create and insert the layered time series.
                layered_name = f"{scheduled_name}_OR_FORECAST"

                layered_ts = forecasted_ts.merge(scheduled_ts, on='DATETIME', how='outer').sort_values("DATETIME")

                layered_ts.loc[:, 'AVGVALUE'] = layered_ts.apply(
                    lambda r: r.AVGVALUE_FORECASTED if np.isnan(r.AVGVALUE_SCHEDULED) else r.AVGVALUE_SCHEDULED,
                    axis=1)
                layered_ts.drop_duplicates(subset=["DATETIME"], inplace=True)

                layered_name = f"{scheduled_name}_OR_FORECAST"
                yes_energy_data[layered_name] = layered_ts[['DATETIME', 'AVGVALUE']]

        # Compute load forecast residuals as features.
        for zone in ("CAISO", "PGE", "SCE", "SDGE", "VEA"):
            load_actual_name = f"{zone}_{DATATYPE_RTLOAD}"
            load_forecast_name = f"{zone}_{DATATYPE_LOAD_FORECAST}"
            load_residual_name = f"{zone}_LOAD_FORECAST_RESIDUAL"
            load_actual_ts = yes_energy_data[load_actual_name].copy()
            load_forecast_ts = yes_energy_data[load_forecast_name].copy()

            load_actual_ts.rename(columns={'AVGVALUE': 'AVGVALUE_ACTUAL'}, inplace=True)
            load_forecast_ts.rename(columns={'AVGVALUE': 'AVGVALUE_FORECASTED'}, inplace=True)

            # load_residual_ts = load_actual_ts.copy()
            load_residual_ts = load_actual_ts.merge(load_forecast_ts, on=['DATETIME', 'TIMEZONE'], how='outer')
            load_residual_ts.AVGVALUE_ACTUAL.ffill(inplace=True)
            load_residual_ts.AVGVALUE_FORECASTED.fillna(load_residual_ts.AVGVALUE_ACTUAL,
                                                        inplace=True)  # fill forecast gap with actual

            # load_residual_ts.fillna(method = 'ffill',inplace= True).fillna(method = 'bfill',inplace= True)

            load_residual_ts.loc[:, 'AVGVALUE'] = load_residual_ts.apply(
                lambda r: r.AVGVALUE_ACTUAL - r.AVGVALUE_FORECASTED,
                axis=1)

            yes_energy_data[load_residual_name] = load_residual_ts[['DATETIME', 'AVGVALUE']]

        # Compute net export: export - import.
        for zone in ('CAISO', 'PGE', 'SCE', 'SDGE'):
            da_exp_name = f'{zone}_{DATATYPE_DA_EXP_MW}'
            da_imp_name = f'{zone}_{DATATYPE_DA_IMP_MW}'
            da_net_exp_name = f'{zone}_DA_NET_EXP_MW'
            da_net_data = define_net_export_data(da_exp_name, da_imp_name, yes_energy_data,
                                                 calendar_ts=calendar_data[self.TIME_ITEMS[0]])
            yes_energy_data.update({da_net_exp_name: da_net_data[['DATETIME', 'AVGVALUE']]})

            rt_exp_name = f'{zone}_{DATATYPE_RT_EXP_MW}'
            rt_imp_name = f'{zone}_{DATATYPE_RT_IMP_MW}'
            rt_net_exp_name = f'{zone}_RT_NET_EXP_MW'
            rt_net_data = define_net_export_data(rt_exp_name, rt_imp_name, yes_energy_data,
                                                 calendar_ts=calendar_data[self.TIME_ITEMS[0]])
            yes_energy_data.update({rt_net_exp_name: rt_net_data[['DATETIME', 'AVGVALUE']]})
        # Compute and clip DART.
        # NOTE: We use tanh-based symmetrical "soft" clipping since it's easier to write as a numerical expression.
        dalmp_ts = yes_energy_data["DALMP"].copy()
        rtlmp_ts = yes_energy_data["RTLMP"].copy()

        dalmp_ts.rename(columns={'AVGVALUE': "AVGVALUE_DA"}, inplace=True)
        rtlmp_ts.rename(columns={'AVGVALUE': "AVGVALUE_RT"}, inplace=True)

        dart_ts = dalmp_ts.merge(rtlmp_ts, on=['DATETIME', 'TIMEZONE'], how='outer')
        dart_ts.ffill(inplace=True)
        dart_ts.loc[:, 'AVGVALUE'] = dart_ts.apply(
            lambda r: r.AVGVALUE_DA - r.AVGVALUE_RT,
            axis=1)

        clipped_dart_ts = dart_ts[['DATETIME', 'AVGVALUE']].copy()
        clipped_dart_ts.AVGVALUE = clipped_dart_ts.AVGVALUE.apply(
            lambda c: self.DART_CLIP * np.tanh(c / self.DART_CLIP))

        yes_energy_data["DART"] = dart_ts[['DATETIME', 'AVGVALUE']]
        yes_energy_data[self.CLIPPED_DART] = clipped_dart_ts

        yes_energy_data = self.apply_time_transformations_on_yes_energy_data(yes_energy_data)

        data = {**yes_energy_data, **calendar_data}
        data_refine = self.apply_timestamp_adjust_on_input_data(data)

        return data_refine

    @property
    def MODELS(self):
        """
        dart_model = lgb.LGBMRegressor(
            objective='mse',
            num_boost_round=225,
            max_depth=10,
            min_child_weight=30,
            learning_rate=0.03,
            subsample=0.6,
            colsample_bytree=0.2,
            #fit_on_null_values=False,
            #predict_on_null_values=False,
            num_leaves = 100)
        """
        params = self.model_parameters
        model_dict = {}
        for model_name in ['DART']:
            model = lgb.LGBMRegressor(**params[model_name])
            model_dict.update({model_name: model})
        return model_dict