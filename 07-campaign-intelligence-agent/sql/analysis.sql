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

-- Age segmentation
WITH segments AS (
    SELECT age, balance, y,
    CASE
        WHEN age IS NULL THEN 'Unknown'
        WHEN age < 30 THEN 'Under 30'
        WHEN age < 45 THEN '30-44'
        WHEN age < 60 THEN '45-59'
        ELSE '60+'
    END AS age_group
    FROM campaign_records
), segment_statistics AS (
    SELECT age_group, COUNT(*) AS total_contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       100.0 * SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)
           AS conversion_rate,
       AVG(age) AS average_age,
       AVG(balance) AS average_balance
    FROM segments
    GROUP BY age_group
    HAVING COUNT(*) >= 1
)
SELECT age_group, total_contacts, subscriptions, non_subscriptions,
       ROUND(conversion_rate, 2) AS conversion_rate,
       ROUND(average_age, 2) AS average_age,
       ROUND(average_balance, 2) AS average_balance
FROM segment_statistics
ORDER BY conversion_rate DESC, total_contacts DESC, age_group;

-- Balance segmentation
WITH segments AS (
    SELECT age, balance, y,
    CASE
        WHEN balance IS NULL THEN 'Unknown'
        WHEN balance < 0 THEN 'Negative'
        WHEN balance = 0 THEN 'Zero'
        WHEN balance < 1000 THEN 'Low'
        WHEN balance < 5000 THEN 'Medium'
        ELSE 'High'
    END AS balance_group
    FROM campaign_records
), segment_statistics AS (
    SELECT balance_group, COUNT(*) AS total_contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       100.0 * SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)
           AS conversion_rate,
       AVG(age) AS average_age,
       AVG(balance) AS average_balance
    FROM segments
    GROUP BY balance_group
    HAVING COUNT(*) >= 1
)
SELECT balance_group, total_contacts, subscriptions, non_subscriptions,
       ROUND(conversion_rate, 2) AS conversion_rate,
       ROUND(average_age, 2) AS average_age,
       ROUND(average_balance, 2) AS average_balance
FROM segment_statistics
ORDER BY conversion_rate DESC, total_contacts DESC, balance_group;

-- Combined age and balance segmentation
WITH age_segments AS (
    SELECT age, balance, y,
    CASE
        WHEN age IS NULL THEN 'Unknown'
        WHEN age < 30 THEN 'Under 30'
        WHEN age < 45 THEN '30-44'
        WHEN age < 60 THEN '45-59'
        ELSE '60+'
    END AS age_group
    FROM campaign_records
), combined_segments AS (
    SELECT age_group, age, balance, y,
    CASE
        WHEN balance IS NULL THEN 'Unknown'
        WHEN balance < 0 THEN 'Negative'
        WHEN balance = 0 THEN 'Zero'
        WHEN balance < 1000 THEN 'Low'
        WHEN balance < 5000 THEN 'Medium'
        ELSE 'High'
    END AS balance_group
    FROM age_segments
), segment_statistics AS (
    SELECT age_group, balance_group, COUNT(*) AS total_contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       100.0 * SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)
           AS conversion_rate,
       AVG(age) AS average_age,
       AVG(balance) AS average_balance
    FROM combined_segments
    GROUP BY age_group, balance_group
    HAVING COUNT(*) >= 100
)
SELECT age_group, balance_group, total_contacts, subscriptions, non_subscriptions,
       ROUND(conversion_rate, 2) AS conversion_rate,
       ROUND(average_age, 2) AS average_age,
       ROUND(average_balance, 2) AS average_balance
FROM segment_statistics
ORDER BY conversion_rate DESC, total_contacts DESC, age_group, balance_group;

-- Top three conversion ranks within each age group
WITH age_segments AS (
    SELECT age, balance, y,
    CASE
        WHEN age IS NULL THEN 'Unknown'
        WHEN age < 30 THEN 'Under 30'
        WHEN age < 45 THEN '30-44'
        WHEN age < 60 THEN '45-59'
        ELSE '60+'
    END AS age_group
    FROM campaign_records
), combined_segments AS (
    SELECT age_group, age, balance, y,
    CASE
        WHEN balance IS NULL THEN 'Unknown'
        WHEN balance < 0 THEN 'Negative'
        WHEN balance = 0 THEN 'Zero'
        WHEN balance < 1000 THEN 'Low'
        WHEN balance < 5000 THEN 'Medium'
        ELSE 'High'
    END AS balance_group
    FROM age_segments
), segment_statistics AS (
    SELECT age_group, balance_group, COUNT(*) AS total_contacts,
       SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscriptions,
       SUM(CASE WHEN y = 'no' THEN 1 ELSE 0 END) AS non_subscriptions,
       100.0 * SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)
           AS conversion_rate,
       AVG(age) AS average_age,
       AVG(balance) AS average_balance
    FROM combined_segments
    GROUP BY age_group, balance_group
    HAVING COUNT(*) >= 100
), ranked_segments AS (
    SELECT *, DENSE_RANK() OVER (
        PARTITION BY age_group ORDER BY conversion_rate DESC
    ) AS segment_rank
    FROM segment_statistics
)
SELECT age_group, balance_group, total_contacts, subscriptions, non_subscriptions,
       ROUND(conversion_rate, 2) AS conversion_rate,
       ROUND(average_age, 2) AS average_age,
       ROUND(average_balance, 2) AS average_balance, segment_rank
FROM ranked_segments
WHERE segment_rank <= 3
ORDER BY age_group, segment_rank, total_contacts DESC, balance_group;
