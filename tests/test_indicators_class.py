"""
Created on Sun Jun  4 01:10:13 2023

@author: Luis Alvaro Correia

Updated: July 27th
    1. Updated function '__cal_SMA()' to receive parameter via 'self'
    2. Fixed the name of function 'test_indicators()'
    3. Adjusted the new name for normalized data as 'data_norm'

"""
import pytest

import numpy as np

from src.finance_ml.data_preparation.data_preparation import DataLoader
from src.finance_ml.indicators.indicators import Indicators

def test_indicators():
    '''
    Tests consistency of Indicators on instantiation and the warning on inconsistent data entry.
    '''
    # Generating a random matrix
    # Defining time_index_col (must be the same column in all inputs) and
    # keep_cols refering to the columns that will remain in the dataset
    dataloader = DataLoader(time_index_col= 'DATE',
                    keep_cols = ['VOLUME','OPEN', 'HIGHT', 'LOW', 'CLOSE', 'VW','TRANSACTIONS'])
    fname_USDBRL = 'FX/USDBRL_2020-04-07_2022-04-06.parquet'

    # No. of Records from example dataset
    N = 10000

    # Dataset chosen in this simulation
    ticker = 'USDBRL'
    fname = fname_USDBRL

    df = dataloader.load_dataset({ticker:'data/'+fname}).iloc[:N]

    # Instanciate the Denoising transformer
    indicator_processor = Indicators(ticker = ticker, norm_data = True)

    assert isinstance(indicator_processor._Indicators__ticker, str)
    assert isinstance(indicator_processor._Indicators__col_open, str)
    assert isinstance(indicator_processor._Indicators__col_high, str)
    assert isinstance(indicator_processor._Indicators__col_low, str)
    assert isinstance(indicator_processor._Indicators__col_close, str)
    assert isinstance(indicator_processor._Indicators__col_volume, str)

    assert isinstance(indicator_processor._Indicators__norm_data, bool)
    assert isinstance(indicator_processor._Indicators__scale_method, str)

    assert isinstance(indicator_processor._Indicators__calc_all, bool)
    assert isinstance(indicator_processor._Indicators__list_ind, list)

    assert isinstance(indicator_processor._Indicators__KAMA_win, int)
    assert isinstance(indicator_processor._Indicators__KAMA_pow1, int)
    assert isinstance(indicator_processor._Indicators__KAMA_pow2, int)
    assert isinstance(indicator_processor._Indicators__PPO_win_slow, int)
    assert isinstance(indicator_processor._Indicators__PPO_win_fast, int)
    assert isinstance(indicator_processor._Indicators__PPO_win_sign, int)
    assert isinstance(indicator_processor._Indicators__PVO_win_slow, int)
    assert isinstance(indicator_processor._Indicators__PVO_win_fast, int)
    assert isinstance(indicator_processor._Indicators__PVO_win_sign, int)
    assert isinstance(indicator_processor._Indicators__ROC_win, int)
    assert isinstance(indicator_processor._Indicators__RSI_win, int)
    assert isinstance(indicator_processor._Indicators__StRSI_win, int)
    assert isinstance(indicator_processor._Indicators__StRSI_sm1, int)
    assert isinstance(indicator_processor._Indicators__StRSI_sm2, int)
    assert isinstance(indicator_processor._Indicators__SO_win, int)
    assert isinstance(indicator_processor._Indicators__SO_sm, int)
    assert isinstance(indicator_processor._Indicators__AOI_win1, int)
    assert isinstance(indicator_processor._Indicators__AOI_win2, int)
    assert isinstance(indicator_processor._Indicators__TSI_win_slow, int)
    assert isinstance(indicator_processor._Indicators__TSI_win_fast, int)
    assert isinstance(indicator_processor._Indicators__UO_win1, int)
    assert isinstance(indicator_processor._Indicators__UO_win2, int)
    assert isinstance(indicator_processor._Indicators__UO_win3, int)
    assert isinstance(indicator_processor._Indicators__UO_weight1, float)
    assert isinstance(indicator_processor._Indicators__UO_weight2, float)
    assert isinstance(indicator_processor._Indicators__UO_weight3, float)
    assert isinstance(indicator_processor._Indicators__WRI_lbp, int)
    assert isinstance(indicator_processor._Indicators__CMF_win, int)
    assert isinstance(indicator_processor._Indicators__EOM_win, int)
    assert isinstance(indicator_processor._Indicators__FI_win, int)
    assert isinstance(indicator_processor._Indicators__MFI_win, int)
    assert isinstance(indicator_processor._Indicators__VWAP_win, int)
    assert isinstance(indicator_processor._Indicators__ADX_win, int)
    assert isinstance(indicator_processor._Indicators__AROON_win, int)
    assert isinstance(indicator_processor._Indicators__CCI_win, int)
    assert isinstance(indicator_processor._Indicators__CCI_const, float)
    assert isinstance(indicator_processor._Indicators__DPO_win, int)
    assert isinstance(indicator_processor._Indicators__EMA_win, int)
    assert isinstance(indicator_processor._Indicators__SMA_win, int)
    assert isinstance(indicator_processor._Indicators__ICHI_win1, int)
    assert isinstance(indicator_processor._Indicators__ICHI_win2, int)
    assert isinstance(indicator_processor._Indicators__ICHI_win3, int)
    assert isinstance(indicator_processor._Indicators__ICHI_visual, bool)
    assert isinstance(indicator_processor._Indicators__KST_roc1, int)
    assert isinstance(indicator_processor._Indicators__KST_roc2, int)
    assert isinstance(indicator_processor._Indicators__KST_roc3, int)
    assert isinstance(indicator_processor._Indicators__KST_roc4, int)
    assert isinstance(indicator_processor._Indicators__KST_win1, int)
    assert isinstance(indicator_processor._Indicators__KST_win2, int)
    assert isinstance(indicator_processor._Indicators__KST_win3, int)
    assert isinstance(indicator_processor._Indicators__KST_win4, int)
    assert isinstance(indicator_processor._Indicators__KST_nsig, int)
    assert isinstance(indicator_processor._Indicators__MACD_win_slow, int)
    assert isinstance(indicator_processor._Indicators__MACD_win_fast, int)
    assert isinstance(indicator_processor._Indicators__MACD_win_sign, int)
    assert isinstance(indicator_processor._Indicators__MI_win_fast, int)
    assert isinstance(indicator_processor._Indicators__MI_win_slow, int)
    assert isinstance(indicator_processor._Indicators__PSAR_step, float)
    assert isinstance(indicator_processor._Indicators__PSAR_max_step, float)
    assert isinstance(indicator_processor._Indicators__STC_win_slow, int)
    assert isinstance(indicator_processor._Indicators__STC_win_fast, int)
    assert isinstance(indicator_processor._Indicators__STC_cycle, int)
    assert isinstance(indicator_processor._Indicators__STC_sm1, int)
    assert isinstance(indicator_processor._Indicators__STC_sm2, int)
    assert isinstance(indicator_processor._Indicators__TRIX_win, int)
    assert isinstance(indicator_processor._Indicators__VI_win, int)
    assert isinstance(indicator_processor._Indicators__WMA_win, int)
    assert isinstance(indicator_processor._Indicators__HMA_win, int)

    df = indicator_processor.fit_transform(df)

    df_norm = indicator_processor.data_norm

    # Test if all columns of normalized data are between -1.0 and 1.0
    for col in df_norm.columns:
        assert np.logical_and(df.filter(regex = col).values >= -1.0,
                          df.filter(regex = col).values <= 1.0).all()

def test_indicators_param_col_open():
    with pytest.raises(ValueError):
        assert Indicators(col_open = 1)

def test_indicators_param_col_high():
    with pytest.raises(ValueError):
        assert Indicators(col_high = 1)

def test_indicators_param_col_low():
    with pytest.raises(ValueError):
        assert Indicators(col_low = 1)

def testindicators_param_col_close():
    with pytest.raises(ValueError):
        assert Indicators(col_close = 1)

def test_indicators_param_col_volume():
    with pytest.raises(ValueError):
        assert Indicators(col_volume = 1)

def test_indicators_param_calc_all():
    with pytest.raises(ValueError):
        assert Indicators(calc_all = 1)

def test_indicators_param_list_ind():
    with pytest.raises(ValueError):
        assert Indicators(calc_all = False,
                          list_ind = ['iNvAlId', 'not VaLIyd'])

def test_indicators_param_norm_data():
	with pytest.raises(ValueError):
		assert Indicators(norm_data = 1.9)

def test_indicators_param_scale_method():
	with pytest.raises(ValueError):
		assert Indicators(scale_method = 'SomeOtherMethod')

def test_indicators_param_KAMA_win():
	with pytest.raises(ValueError):
		assert Indicators(KAMA_win = -5.06482912850792)

def test_indicators_param_KAMA_pow1():
    with pytest.raises(ValueError):
        assert Indicators(KAMA_pow1 = -4.39733654041853)

def test_indicators_param_KAMA_pow2():
    with pytest.raises(ValueError):
        assert Indicators(KAMA_pow2 = -3.49832193692309)

def test_indicators_param_PPO_win_slow():
    with pytest.raises(ValueError):
        assert Indicators(PPO_win_slow = -4.8633816380185)

def test_indicators_param_PPO_win_fast():
    with pytest.raises(ValueError):
        assert Indicators(PPO_win_fast = -1.9967934720844)

def test_indicators_param_PPO_win_sign():
    with pytest.raises(ValueError):
        assert Indicators(PPO_win_sign = -5.9019074462193)

def test_indicators_param_PVO_win_slow():
    with pytest.raises(ValueError):
        assert Indicators(PVO_win_slow = -6.84902552247935)

def test_indicators_param_PVO_win_fast():
    with pytest.raises(ValueError):
        assert Indicators(PVO_win_fast = -3.01213773111448)

def test_indicators_param_PVO_win_sign():
    with pytest.raises(ValueError):
        assert Indicators(PVO_win_sign = -2.89565071278062)

def test_indicators_param_ROC_win():
    with pytest.raises(ValueError):
        assert Indicators(ROC_win = -3.94725436160503)

def test_indicators_param_RSI_win():
    with pytest.raises(ValueError):
        assert Indicators(RSI_win = -5.21039842892421)

def test_indicators_param_StRSI_win():
    with pytest.raises(ValueError):
        assert Indicators(StRSI_win = -6.2138108232211)

def test_indicators_param_StRSI_sm1():
    with pytest.raises(ValueError):
        assert Indicators(StRSI_sm1 = -2.21886313395992)

def test_indicators_param_StRSI_sm2():
    with pytest.raises(ValueError):
        assert Indicators(StRSI_sm2 = -5.6703343850065)

def test_indicators_param_SO_win():
    with pytest.raises(ValueError):
        assert Indicators(SO_win = -2.05227800096332)

def test_indicators_param_SO_sm():
    with pytest.raises(ValueError):
        assert Indicators(SO_sm = -4.43915718437036)

def test_indicators_param_AOI_win1():
    with pytest.raises(ValueError):
        assert Indicators(AOI_win1 = -5.32602722895092)

def test_indicators_param_AOI_win2():
    with pytest.raises(ValueError):
        assert Indicators(AOI_win2 = -6.46884804390941)

def test_indicators_param_TSI_win_slow():
    with pytest.raises(ValueError):
        assert Indicators(TSI_win_slow = -3.6258197705552)

def test_indicators_param_TSI_win_fast():
    with pytest.raises(ValueError):
        assert Indicators(TSI_win_fast = -2.57246280560111)

def test_indicators_param_UO_win1():
    with pytest.raises(ValueError):
        assert Indicators(UO_win1 = -2.37910244159106)

def test_indicators_param_UO_win2():
    with pytest.raises(ValueError):
        assert Indicators(UO_win2 = -5.05131442115531)

def test_indicators_param_UO_win3():
    with pytest.raises(ValueError):
        assert Indicators(UO_win3 = -2.06192362307759)

def test_indicators_param_UO_weight1():
    with pytest.raises(ValueError):
        assert Indicators(UO_weight1 = -3.5428473255669)

def test_indicators_param_UO_weight2():
    with pytest.raises(ValueError):
        assert Indicators(UO_weight2 = -2.96442191730946)

def test_indicators_param_UO_weight3():
    with pytest.raises(ValueError):
        assert Indicators(UO_weight3 = -1.29231196949221)

def test_indicators_param_WRI_lbp():
    with pytest.raises(ValueError):
        assert Indicators(WRI_lbp = -1.56024304875537)

def test_indicators_param_CMF_win():
    with pytest.raises(ValueError):
        assert Indicators(CMF_win = -5.4293896260703)

def test_indicators_param_EOM_win():
    with pytest.raises(ValueError):
        assert Indicators(EOM_win = -2.71270499718407)

def test_indicators_param_FI_win():
    with pytest.raises(ValueError):
        assert Indicators(FI_win = -5.09729550031399)

def test_indicators_param_MFI_win():
    with pytest.raises(ValueError):
        assert Indicators(MFI_win = -1.99542483195893)

def test_indicators_param_VWAP_win():
    with pytest.raises(ValueError):
        assert Indicators(VWAP_win = -3.67897573927379)

def test_indicators_param_ADX_win():
    with pytest.raises(ValueError):
        assert Indicators(ADX_win = -1.50872035724418)

def test_indicators_param_AROON_win():
    with pytest.raises(ValueError):
        assert Indicators(AROON_win = -1.95105423796558)

def test_indicators_param_CCI_win():
    with pytest.raises(ValueError):
        assert Indicators(CCI_win = -4.06790858758707)

def test_indicators_param_CCI_const():
    with pytest.raises(ValueError):
        assert Indicators(CCI_const = -5.63556942252038)

def test_indicators_param_DPO_win():
    with pytest.raises(ValueError):
        assert Indicators(DPO_win = -3.22173591407414)

def test_indicators_param_EMA_win():
    with pytest.raises(ValueError):
        assert Indicators(EMA_win = -5.55577474919645)

def test_indicators_param_SMA_win():
    with pytest.raises(ValueError):
        assert Indicators(SMA_win = -4.5588639645)

def test_indicators_param_ICHI_win1():
    with pytest.raises(ValueError):
        assert Indicators(ICHI_win1 = -6.16373560919093)

def test_indicators_param_ICHI_win2():
    with pytest.raises(ValueError):
        assert Indicators(ICHI_win2 = -2.18726169007601)

def test_indicators_param_ICHI_win3():
    with pytest.raises(ValueError):
        assert Indicators(ICHI_win3 = -2.75148019347847)

def test_indicators_param_ICHI_visual():
    with pytest.raises(ValueError):
        assert Indicators(ICHI_visual = -4.49344668617071)

def test_indicators_param_KST_roc1():
    with pytest.raises(ValueError):
        assert Indicators(KST_roc1 = -4.63668069610687)

def test_indicators_param_KST_roc2():
    with pytest.raises(ValueError):
        assert Indicators(KST_roc2 = -2.43180552592803)

def test_indicators_param_KST_roc3():
    with pytest.raises(ValueError):
        assert Indicators(KST_roc3 = -4.92350305226444)

def test_indicators_param_KST_roc4():
    with pytest.raises(ValueError):
        assert Indicators(KST_roc4 = -2.351854774353)

def test_indicators_param_KST_win1():
    with pytest.raises(ValueError):
        assert Indicators(KST_win1 = -1.46596670961599)

def test_indicators_param_KST_win2():
    with pytest.raises(ValueError):
        assert Indicators(KST_win2 = -5.7966666849707)

def test_indicators_param_KST_win3():
    with pytest.raises(ValueError):
        assert Indicators(KST_win3 = -1.42060689636332)

def test_indicators_param_KST_win4():
    with pytest.raises(ValueError):
        assert Indicators(KST_win4 = -4.15821804464781)

def test_indicators_param_KST_nsig():
    with pytest.raises(ValueError):
        assert Indicators(KST_nsig = -4.2825140204499)

def test_indicators_param_MACD_win_slow():
    with pytest.raises(ValueError):
        assert Indicators(MACD_win_slow = -2.10647580110654)

def test_indicators_param_MACD_win_fast():
    with pytest.raises(ValueError):
        assert Indicators(MACD_win_fast = -1.05272335592645)

def test_indicators_param_MACD_win_sign():
    with pytest.raises(ValueError):
        assert Indicators(MACD_win_sign = -1.47804411273431)

def test_indicators_param_MI_win_fast():
    with pytest.raises(ValueError):
        assert Indicators(MI_win_fast = -2.77375354822435)

def test_indicators_param_MI_win_slow():
    with pytest.raises(ValueError):
        assert Indicators(MI_win_slow = -6.44479603672348)

def test_indicators_param_PSAR_step():
    with pytest.raises(ValueError):
        assert Indicators(PSAR_step = -2.46192097030094)

def test_indicators_param_PSAR_max_step():
    with pytest.raises(ValueError):
        assert Indicators(PSAR_max_step = -2.20777698139329)

def test_indicators_param_STC_win_slow():
    with pytest.raises(ValueError):
        assert Indicators(STC_win_slow = -6.85111123282612)

def test_indicators_param_STC_win_fast():
    with pytest.raises(ValueError):
        assert Indicators(STC_win_fast = -5.94090107759065)

def test_indicators_param_STC_cycle():
    with pytest.raises(ValueError):
        assert Indicators(STC_cycle = -3.87253800771118)

def test_indicators_param_STC_sm1():
    with pytest.raises(ValueError):
        assert Indicators(STC_sm1 = -3.91392520527165)

def test_indicators_param_STC_sm2():
    with pytest.raises(ValueError):
        assert Indicators(STC_sm2 = -5.56313805554748)

def test_indicators_param_TRIX_win():
    with pytest.raises(ValueError):
        assert Indicators(TRIX_win = -2.78539038164447)

def test_indicators_param_VI_win():
    with pytest.raises(ValueError):
        assert Indicators(VI_win = -5.1863866785022)

def test_indicators_param_WMA_win():
    with pytest.raises(ValueError):
        assert Indicators(WMA_win = -5.26755227536053)

def test_indicators_param_HMA_win():
    with pytest.raises(ValueError):
        assert Indicators(HMA_win = -5.26755227536053)


