-- Run with: sqlite3 data/campaigns.sqlite3 < sql/analysis.sql
.headers on
.mode column

-- Campaign overview
SELECT COUNT(*) AS total_records,
       COALESCE(SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END), 0) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent
FROM campaign_records;

-- Group by job
SELECT job, COUNT(*) AS total_records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent,
       ROUND(AVG(balance), 2) AS average_balance
FROM campaign_records
GROUP BY job
HAVING COUNT(*) >= 1
ORDER BY subscription_rate_percent DESC, total_records DESC, job;

-- Group by education
SELECT education, COUNT(*) AS total_records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent,
       ROUND(AVG(balance), 2) AS average_balance
FROM campaign_records
GROUP BY education
HAVING COUNT(*) >= 1
ORDER BY subscription_rate_percent DESC, total_records DESC, education;

-- Group by poutcome
SELECT poutcome, COUNT(*) AS total_records,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS subscription_rate_percent,
       ROUND(AVG(balance), 2) AS average_balance
FROM campaign_records
GROUP BY poutcome
HAVING COUNT(*) >= 1
ORDER BY subscription_rate_percent DESC, total_records DESC, poutcome;

-- Previously contacted segments with positive balances
SELECT job, poutcome, COUNT(*) AS total_contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       ROUND(100.0 * AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END), 2)
           AS conversion_rate,
       ROUND(AVG(balance), 2) AS average_balance
FROM campaign_records
WHERE balance > 0 AND pdays != -1
GROUP BY job, poutcome
HAVING COUNT(*) >= 50
ORDER BY conversion_rate DESC, total_contacts DESC, job, poutcome
LIMIT 10;
