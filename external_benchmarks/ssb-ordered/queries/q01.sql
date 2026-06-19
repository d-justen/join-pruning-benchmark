SELECT d_year, sum(lo_revenue) 
FROM lineorder
JOIN date ON (lo_orderdate = d_datekey)
WHERE d_month = 'November'
GROUP BY d_year;

