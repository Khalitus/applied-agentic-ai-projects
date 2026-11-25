-- Run using: sqlite3 data/campaigns.sqlite3 < sql/analysis.sql
.headers on
.mode column

SELECT COUNT(*) AS total_records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscribed,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2) AS rate_pct
FROM campaign_records;

SELECT job, COUNT(*) AS customers,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2) AS rate_pct
FROM campaign_records
GROUP BY job
HAVING COUNT(*) >= 100
ORDER BY rate_pct DESC;
