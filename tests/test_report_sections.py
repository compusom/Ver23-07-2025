import pandas as pd
from datetime import datetime
from data_processing.report_sections import (
    _generar_tabla_bitacora_top_ads,
    _generar_tabla_bitacora_top_adsets,
    _generar_tabla_bitacora_top_campaigns,
    _generar_tabla_bitacora_top_entities,
    METRIC_LABELS_ADS,
    METRIC_LABELS_BASE,
    _generar_tabla_performance_publico,
    _generar_tabla_tendencia_ratios
)
from data_processing.report_sections import _clean_audience_string


def test_top_ads_basic_columns(capsys):
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01','2024-06-02']),
        'Campaign': ['Camp','Camp'],
        'AdSet': ['Set','Set'],
        'Anuncio': ['Ad1','Ad1'],
        'spend': [10, 20],
        'impr': [100, 200],
        'reach': [80, 150],
        'purchases': [1, 2],
        'visits': [10, 20],
        'value': [50, 120],
        'clicks': [5, 10],
        'clicks_out': [1, 2],
        'rv3': [0, 0],
        'rv25': [1, 2],
        'rv75': [0, 1],
        'rv100': [0, 1],
        'rtime': [4, 5],
        'url_final': ['https://ex.com', 'https://ex.com'],
        'puja': [0.5, 0.5],
        'interacciones': [7, 8],
        'comentarios': [1, 2],
        'Públicos In': ['Inc1', 'Inc2'],
        'Públicos Ex': ['Exc1', 'Exc2'],
    })
    periods = [
        (datetime(2024,6,1), datetime(2024,6,2), 'Semana actual'),
        (datetime(2024,5,25), datetime(2024,5,31), '1ª semana anterior'),
        (datetime(2024,5,18), datetime(2024,5,24), '2ª semana anterior'),
    ]
    active = pd.DataFrame({
        'Campaign': ['Camp'],
        'AdSet': ['Set'],
        'Anuncio': ['Ad1'],
        'Días_Activo_Total': [2]
    })
    logs = []
    _generar_tabla_bitacora_top_ads(df, periods, active, logs.append, '$', top_n=1)
    output = "\n".join(logs)
    assert 'Top 1 Ads Bitácora - Semana actual' in output
    assert 'Anuncio' in output
    assert 'Días Act' in output
    assert 'Ventas' in output
    assert 'RV25%' in output
    assert 'RV75%' in output
    assert 'RV100%' in output
    assert 'Tiempo RV (s)' in output

def test_clean_audience_string():
    assert _clean_audience_string('123:Aud1 | 456:Aud2') == 'Aud1, Aud2'
    # Also handle comma separated values
    assert _clean_audience_string('123:Aud1, 456:Aud2') == 'Aud1, Aud2'
    # Should detect new audiences separated only by spaces
    sample = '120219213417610120:IG Engagers 60d 120982364718293172:Web Visitors 30d'
    expected = 'IG Engagers 60d, Web Visitors 30d'
    assert _clean_audience_string(sample) == expected


def test_top_adsets_weekly_table(capsys):
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01', '2024-06-02']),
        'Campaign': ['Camp', 'Camp'],
        'AdSet': ['Set', 'Set'],
        'spend': [10, 15],
        'impr': [100, 150],
        'reach': [90, 130],
        'purchases': [1, 1],
        'visits': [10, 12],
        'value': [20, 25],
    })
    periods = [
        (datetime(2024, 6, 1), datetime(2024, 6, 2), 'Semana actual'),
        (datetime(2024, 5, 25), datetime(2024, 5, 31), '1ª semana anterior'),
    ]
    active = pd.DataFrame({
        'Campaign': ['Camp'],
        'AdSet': ['Set'],
        'Días_Activo_Total': [2]
    })
    logs = []
    _generar_tabla_bitacora_top_adsets(df, periods, active, logs.append, '$', top_n=1)
    output = "\n".join(logs)
    assert 'Top 1 AdSets Bitácora - Semana actual' in output
    assert 'Días Act' in output
    assert 'Ventas' in output


def test_top_adsets_deduplication():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01', '2024-06-02']),
        'Campaign': ['Camp', 'Camp'],
        'AdSet': ['Set', 'Set'],
        'spend': [10, 15],
        'impr': [100, 150],
        'reach': [90, 130],
        'purchases': [1, 1],
        'visits': [10, 12],
        'value': [20, 25],
    })
    periods = [(datetime(2024, 6, 1), datetime(2024, 6, 2), 'Semana actual')]
    # Active days with duplicate row
    active = pd.DataFrame({
        'Campaign': ['Camp', 'Camp'],
        'AdSet': ['Set', 'Set'],
        'Días_Activo_Total': [2, 2],
    })
    logs = []
    _generar_tabla_bitacora_top_adsets(df, periods, active, logs.append, '$', top_n=1)
    row_lines = [l for l in logs if l.startswith('| Camp ')]
    assert len(row_lines) == 1


def test_top_adsets_deduplication():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01', '2024-06-02']),
        'Campaign': ['Camp', 'Camp'],
        'AdSet': ['Set', 'Set'],
        'spend': [10, 15],
        'impr': [100, 150],
        'reach': [90, 130],
        'purchases': [1, 1],
        'visits': [10, 12],
        'value': [20, 25],
    })
    periods = [(datetime(2024, 6, 1), datetime(2024, 6, 2), 'Semana actual')]
    # Active days with duplicate row
    active = pd.DataFrame({
        'Campaign': ['Camp', 'Camp'],
        'AdSet': ['Set', 'Set'],
        'Días_Activo_Total': [2, 2],
    })
    logs = []
    _generar_tabla_bitacora_top_adsets(df, periods, active, logs.append, '$', top_n=1)
    row_lines = [l for l in logs if l.startswith('| Camp ')]
    assert len(row_lines) == 1


def test_top_campaigns_weekly_table(capsys):
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01', '2024-06-02']),
        'Campaign': ['Camp', 'Camp'],
        'spend': [10, 15],
        'impr': [100, 150],
        'reach': [90, 130],
        'purchases': [1, 1],
        'visits': [10, 12],
        'value': [20, 25],
    })
    periods = [
        (datetime(2024, 6, 1), datetime(2024, 6, 2), 'Semana actual'),
        (datetime(2024, 5, 25), datetime(2024, 5, 31), '1ª semana anterior'),
    ]
    active = pd.DataFrame({
        'Campaign': ['Camp'],
        'Días_Activo_Total': [2]
    })
    logs = []
    _generar_tabla_bitacora_top_campaigns(df, periods, active, logs.append, '$', top_n=1)
    output = "\n".join(logs)
    assert 'Top 1 Campañas Bitácora - Semana actual' in output
    assert 'Ventas' in output


def test_generic_helper_ads(capsys):
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01','2024-06-02']),
        'Campaign': ['Camp','Camp'],
        'AdSet': ['Set','Set'],
        'Anuncio': ['Ad1','Ad1'],
        'spend': [10, 20],
        'impr': [100, 200],
        'reach': [80, 150],
        'purchases': [1, 2],
        'visits': [10, 20],
        'value': [50, 120],
    })
    periods = [
        (datetime(2024,6,1), datetime(2024,6,2), 'Semana actual')
    ]
    active = pd.DataFrame({
        'Campaign': ['Camp'],
        'AdSet': ['Set'],
        'Anuncio': ['Ad1'],
        'Días_Activo_Total': [2]
    })
    logs = []
    _generar_tabla_bitacora_top_entities(
        df,
        periods,
        active,
        logs.append,
        '$',
        ['Campaign','AdSet','Anuncio'],
        'Ads',
        METRIC_LABELS_ADS,
        ranking_method='ads',
        top_n=1
    )
    output = "\n".join(logs)
    assert 'Top 1 Ads Bitácora - Semana actual' in output


def test_performance_publico_table():
    df = pd.DataFrame({
        'Públicos In': ['Aud1', 'Aud1', 'Aud2'],
        'spend': [10, 15, 5],
        'purchases': [1, 2, 0],
        'value': [20, 40, 0],
        'impr': [100, 200, 50],
        'clicks': [5, 10, 2],
        'reach': [80, 150, 30],
        'date': pd.to_datetime(['2024-06-01', '2024-06-02', '2024-06-01']),
    })
    logs = []
    _generar_tabla_performance_publico(df, logs.append, '$', top_n=2)
    output = "\n".join(logs)
    assert 'TABLA: PERFORMANCE_PUBLICO' in output


def test_tendencia_ratios_weekly():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-06-01', '2024-06-02', '2024-05-26']),
        'clicks': [10, 5, 4],
        'impr': [100, 80, 50],
        'visits': [20, 15, 10],
        'addcart': [5, 3, 2],
        'checkout': [2, 1, 1],
        'purchases': [1, 0, 1],
    })
    periods = [
        (datetime(2024, 6, 1), datetime(2024, 6, 2), 'Semana actual'),
        (datetime(2024, 5, 26), datetime(2024, 5, 26), '1ª semana anterior'),
    ]
    logs = []
    _generar_tabla_tendencia_ratios(df, periods, logs.append, period_type='Weeks')
    output = "\n".join(logs)
    assert 'TABLA: TENDENCIA_RATIOS' in output

