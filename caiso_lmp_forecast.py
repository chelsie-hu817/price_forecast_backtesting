# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:50:02 2023

@author: lzhao
"""
import pandas as pd
import numpy as np
from dateutil import relativedelta
import datetime

import lightgbm as lgb

from caiso_price_forecast import caiso_price_forecast_base
from caiso_price_forecast import DATATYPE_DALMP, DATATYPE_RTLMP
from caiso_price_forecast import DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, DATATYPE_LOAD_FORECAST_2DAY, DATATYPE_RTLOAD, \
    DATATYPE_SOLAR_FORECAST, DATATYPE_WIND_DA, DATATYPE_WIND_FORECAST

DATATYPE_DA_EXP_MW = 'DA_EXP_MW'
DATATYPE_DA_IMP_MW = 'DA_IMP_MW'
DATATYPE_RT_EXP_MW = 'RT_EXP_MW'
DATATYPE_RT_IMP_MW = 'RT_IMP_MW'

def define_net_export_data(exp_name, import_name, yes_energy_data, calendar_ts):
    """

    :param exp_name: string, name of export data ts
    :param import_name: string, name of import data ts
    :param yes_energy_data: dict, dictionary of yes energy data
    :param calendar_ts: pd.DataFrame, dataframe with full time period
    :return: pd.DataFrame
    """
    exp_data = yes_energy_data[exp_name].copy()
    imp_data = yes_energy_data[import_name].copy()

    exp_data.rename(columns={'AVGVALUE': 'AVGVALUE_EXP'}, inplace=True)
    imp_data.rename(columns={'AVGVALUE': 'AVGVALUE_IMP'}, inplace=True)

    # load_residual_ts = load_actual_ts.copy()
    net_exp = exp_data.merge(imp_data, on=['DATETIME', 'TIMEZONE'], how='outer')
    net_exp.ffill(inplace=True)

    net_exp.loc[:, 'AVGVALUE'] = net_exp.apply(
        lambda r: r.AVGVALUE_EXP - r.AVGVALUE_IMP,
        axis=1)
    net_exp = calendar_ts[['DATETIME']].merge(net_exp, on='DATETIME', how='left').ffill()
    net_exp.drop_duplicates(subset=["DATETIME", "TIMEZONE"], inplace=True)
    return net_exp


class caiso_lmp_da_forecast(caiso_price_forecast_base):
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
        caiso lmp forecast client (at da level)

        """
        super().__init__(name)
        self.PRICE_NODE = price_node
        self._LOCAL_WEATHER_STATIONS = None

    @property
    def LOCAL_WEATHER_STATIONS(self):
        return self._LOCAL_WEATHER_STATIONS

    @LOCAL_WEATHER_STATIONS.setter
    def LOCAL_WEATHER_STATIONS(self, value):
        if isinstance(value, str) and value not in self._WEATHER_STATIONS:
            self._LOCAL_WEATHER_STATIONS = value
        elif isinstance(value, list):
            self._LOCAL_WEATHER_STATIONS = [c for c in value if c not in self._WEATHER_STATIONS]

    @property
    def WEATHER_STATIONS(self):
        if self.LOCAL_WEATHER_STATIONS is None or len(self.LOCAL_WEATHER_STATIONS) == 0:
            return self._WEATHER_STATIONS
        else:
            if isinstance(self.LOCAL_WEATHER_STATIONS, str):
                return self._WEATHER_STATIONS + [self.LOCAL_WEATHER_STATIONS]
            elif isinstance(self.LOCAL_WEATHER_STATIONS, list):
                return self._WEATHER_STATIONS + self.LOCAL_WEATHER_STATIONS
            else:
                return self._WEATHER_STATIONS

    @WEATHER_STATIONS.setter
    def WEATHER_STATIONS(self, weather_stations_list):
        super(caiso_lmp_da_forecast, type(self)).WEATHER_STATIONS.fset(self, weather_stations_list)

    @property
    def DART_CLIP(self):
        return 270

    @property
    def CLIPPED_DART(self):
        return f"DART_SOFT_CLIPPED_{self.DART_CLIP:.0f}"

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
            # LMP
            "DALMP": (DATATYPE_DALMP, PRICE_NODE),
            "RTLMP": (DATATYPE_RTLMP, PRICE_NODE),

            # Load forecasts.
            "CAISO_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_CAISO),
            "PGE_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_PGE),
            "SCE_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_SCE),
            "SDGE_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_SDGE),
            "VEA_LOAD_FORECAST_2DAY": (DATATYPE_LOAD_FORECAST_2DAY, ZONE_VEA),
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
            # Forecasted renewable generation.
            "NP15_SOLAR_FORECAST": (DATATYPE_SOLAR_FORECAST, ZONE_NP15),
            "NP15_WIND_FORECAST": (DATATYPE_WIND_FORECAST, ZONE_NP15),
            "SP15_SOLAR_FORECAST": (DATATYPE_SOLAR_FORECAST, ZONE_SP15),
            "SP15_WIND_FORECAST": (DATATYPE_WIND_FORECAST, ZONE_SP15),

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
            'SDGE_RT_EXP_MW': (DATATYPE_RT_EXP_MW, INTERFACE_SDGE)

        }

        return YES_ENERGY_ITEMS

    @property
    def BASE_FEATURES(self):
        base_features = [
            "CAISO_LOAD_FORECAST_2DAY",
            "PGE_LOAD_FORECAST_2DAY",
            "SCE_LOAD_FORECAST_2DAY",
            "SDGE_LOAD_FORECAST_2DAY",
            "VEA_LOAD_FORECAST_2DAY",
            "NP15_GENERATION_SOLAR_DA_SCHEDULE_OR_FORECAST",
            "NP15_WIND_DA_OR_FORECAST",
            "SP15_GENERATION_SOLAR_DA_SCHEDULE_OR_FORECAST",
            "SP15_WIND_DA_OR_FORECAST",
        ]
        return base_features

    @property
    def YES_ENERGY_TIME_TRANSFORMATIONS(self):
        _base_transformation = super().YES_ENERGY_TIME_TRANSFORMATIONS
        _base_transformation.update(
            {
                **{f'{zone}_DA_NET_EXP_MW': [24, 48, 168] for zone in ['CAISO', 'PGE', 'SCE', 'SDGE']},
                **{f'{zone}_RT_NET_EXP_MW': [48, 168] for zone in ['CAISO', 'PGE', 'SCE', 'SDGE']},

            }
        )
        return _base_transformation

    @property
    def TARGETS(self):
        return ['DALMP', self.CLIPPED_DART]

    def define_data(self, yes_energy_data):
        calendar_ts = yes_energy_data['DALMP'][['DATETIME']].copy()  ##use dalmp as example
        calendar_data = self.define_calendar_data(calendar_ts)

        # Since the scheduled datatypes may not be available in time for the model to use them,
        #   we use layering to set up the corresponding forecasts as a backup.
        for zone in ("NP15", "SP15"):
            for scheduled, forecasted in (
                    (DATATYPE_GENERATION_SOLAR_DA_SCHEDULE, DATATYPE_SOLAR_FORECAST),
                    (DATATYPE_WIND_DA, DATATYPE_WIND_FORECAST),
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
            load_forecast_name = f"{zone}_{DATATYPE_LOAD_FORECAST_2DAY}"
            load_residual_name = f"{zone}_LOAD_FORECAST_RESIDUAL"
            load_residual_ts = self.define_load_residual_data(load_actual_name, load_forecast_name, yes_energy_data,
                                                              calendar_ts=calendar_data[self.TIME_ITEMS[0]])

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
        dart_ts.ffill(inplace=True)
        dart_ts.loc[:, 'AVGVALUE'] = dart_ts.apply(
            lambda r: r.AVGVALUE_DA - r.AVGVALUE_RT, axis=1)

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
    def model_parameters(self):
        da_params = {"objective": "mae",
                     "num_boost_round": 800,
                     "max_depth": 8,
                     "min_child_weight": 25,
                     "learning_rate": 0.09,
                     "num_leaves": 80,
                     }

        dart_params = {"objective": "mse",
                       "num_boost_round": 225,
                       "max_depth": 10,
                       "min_child_weight": 30,
                       "learning_rate": 0.03,
                       "subsample": 0.6,
                       "colsample_bytree": 0.2,
                       "num_leaves": 100,
                       }
        return {'DA': da_params, 'DART': dart_params}

    @property
    def MODELS(self):
        params = self.model_parameters
        """
        da_model = lgb.LGBMRegressor(
            objective='mae',
            num_iterations=800,
            max_depth=8,
            min_child_weight=25,
            learning_rate=0.09,
            num_leaves = 80)

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
        model_dict = {}
        for model_name in ['DA', 'DART']:
            model = lgb.LGBMRegressor(**params[model_name])
            model_dict.update({model_name: model})
        return model_dict

    def percentile_models(self, forecast_type, p):
        """
        :param forecast_type: string, DA/DART
        :param p: percentile (0-1), float or list
        :return: dict of model to generate percentile
        """
        assert self.model_parameters.get(
            forecast_type) is not None, f"{forecast_type} model parameters are not defined."
        param = self.model_parameters[forecast_type]

        model_dict = {}
        if isinstance(p, list):
            for _p in p:
                _param = param.copy()
                _param.update({'objective': 'quantile',
                               'alpha': _p
                               })
                model = lgb.LGBMRegressor(**_param)
                model_dict.update({f'{forecast_type}_p{_p * 100:.0f}': model})
        else:
            _param = param.copy()
            _param.update({'objective': 'quantile',
                           'alpha': p
                           })
            model = lgb.LGBMRegressor(**_param)
            model_dict.update({f'{forecast_type}_p{p * 100:.0f}': model})
        return model_dict


# =========== Energy Price Forecasting =======================
class caiso_energy_da_forecast(caiso_lmp_da_forecast):
    """
    ENERGY day-ahead forecasting with ENERGY-native price names,
    but uses the SAME spread name as LMP: DART_SOFT_CLIPPED_*.
    """

    DATATYPE_DAENERGY = "DAENERGY"
    DATATYPE_RTENERGY = "RTENERGY"

    def __init__(self, name, price_node, dart_clip: float = 270, **kwargs):        # Clip size changeable
        super().__init__(name = name, price_node = price_node, **kwargs)
        self._dart_clip_energy = float(dart_clip)

    @property
    def DART_CLIP(self):
        return self._dart_clip_energy

    # Targets: DAENERGY and (ENERGY-based) DART, same name as base
    @property
    def TARGETS(self):
        return ["DAENERGY", self.CLIPPED_DART]  # keep DART naming

    @property
    def YES_ENERGY_ITEMS(self):
        """Return item map with ENERGY keys (prices only)."""
        items = super().YES_ENERGY_ITEMS.copy()
        items.pop("DALMP", None)
        items.pop("RTLMP", None)
        items["DAENERGY"] = (self.DATATYPE_DAENERGY, self.PRICE_NODE)
        items["RTENERGY"] = (self.DATATYPE_RTENERGY, self.PRICE_NODE)
        return items

    @staticmethod
    def _rename_cols_prices_to_energy(df):
        """Rename only price-related column names (keep DART as-is)."""
        rename_map = {}
        for c in df.columns:
            new_c = c.replace("DALMP", "DAENERGY").replace("RTLMP", "RTENERGY")
            if new_c != c:
                rename_map[c] = new_c
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        return df

    def define_data(self, yes_energy_data):
        """
        Alias DAENERGY/RTENERGY to DALMP/RTLMP for the parent pipeline,
        then rename price keys/columns back to ENERGY names. Keep DART names.
        """
        if "DAENERGY" not in yes_energy_data or "RTENERGY" not in yes_energy_data:
            raise KeyError("Expected 'DAENERGY' and 'RTENERGY' in yes_energy_data.")

        temp = dict(yes_energy_data)
        temp["DALMP"] = yes_energy_data["DAENERGY"]
        temp["RTLMP"] = yes_energy_data["RTENERGY"]

        parent_data = super().define_data(temp)  # uses DALMP/RTLMP & builds DART, lags, etc

        out = {}
        for k, v in parent_data.items():
            new_key = k.replace("DALMP", "DAENERGY").replace("RTLMP", "RTENERGY")
            df = v.copy()
            self._rename_cols_prices_to_energy(df)
            out[new_key] = df


        return out

