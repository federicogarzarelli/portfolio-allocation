DELETE FROM FACT_HISTPRICES WHERE ticker IN (SELECT ticker FROM DIM_STOCKS WHERE asset_class IN ('not_categorized','FX'));
DELETE FROM DIM_STOCKS WHERE asset_class IN ('not_categorized','FX');
